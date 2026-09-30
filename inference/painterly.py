"""Stage 4 - Hertzmann-style curved brush stroke extraction.

Implements the curved-brush-stroke painterly algorithm
( Hertzmann, "Painterly Rendering with Curved Brush Strokes of Different
Sizes", SIGGRAPH 1998 ) adapted for the WebGL renderer:

* per semantic layer, a stack of brush radii is painted coarse-to-fine
  ( the sky gets long sweeping strokes, the village fine detail dabs );
* each stroke starts from a jittered grid seed inside the layer mask and
  follows the direction **perpendicular to the local image gradient** -
  i.e. along iso-luminance contours, which is what produces the curved,
  Van Gogh-like flow of strokes;
* colours are sampled from the *original* painting ( preservation ): the
  renderer never invents palette values, it only moves them.

The stroke list is serialised to a flat ``float32`` binary with a fixed
record stride of 12 floats per segment, ready for GPU instanced rendering
( the three vec4 attributes of ``brush.vert`` )::

    [ x, y, dir_x, dir_y,            # aSegA: center + direction
      length, radius, opacity,       # aSegB: dab geometry + paint
      layer_idx, r, g, b,            # aSegC halves: layer + colour
      phase ]                        # ( aSegB.w = layer, aSegC = rgb+phase )

The grouping MUST match the attribute declaration order in
``renderer/shaders/brush.vert`` ( aSegA, aSegB, aSegC ).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import utils
from .utils import ensure_dir, get_logger, atomic_write_json

LOGGER = get_logger()

SEGMENT_STRIDE = 12

# brush radius stacks per semantic layer ( px at 768-long-side scale )
LAYER_SCALES: Dict[str, List[float]] = {
    "sky": [30.0, 15.0, 7.5],
    "mountains": [20.0, 10.0],
    "village": [14.0, 7.0, 3.5],
    "tree": [16.0, 8.0, 4.0],
    "stars": [12.0, 6.0, 3.0],
    "moon": [12.0, 6.0, 3.0],
}
DEFAULT_SCALES = [18.0, 9.0, 4.5]

# scale radii with processing resolution ( reference long side = 768 )
REFERENCE_SIDE = 768.0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _luminance(rgb01: np.ndarray) -> np.ndarray:
    return rgb01 @ np.array([0.299, 0.587, 0.114], np.float32)


def _field_gradient(lum_blur: np.ndarray):
    import cv2

    gx = cv2.Sobel(lum_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum_blur, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _iso_direction(gx: np.ndarray, gy: np.ndarray, x: int, y: int,
                   fallback: np.ndarray) -> np.ndarray:
    """Direction perpendicular to the gradient ( along iso-contours )."""
    g = np.array([gy[y, x], -gx[y, x]], np.float32)  # rotated 90 deg
    norm = float(np.hypot(*g))
    if norm < 1e-3:
        return fallback
    return g / norm


def _grow_chain(seed: Tuple[float, float], dir0: np.ndarray, mask: np.ndarray,
                gx: np.ndarray, gy: np.ndarray, radius: float) -> List[Tuple[float, float]]:
    """Grow a curved stroke both ways from the seed along iso-contours."""
    h, w = mask.shape
    step = max(radius * 0.5, 1.6)
    # longer chains than the original: sweeping curved strokes read far more
    # Van Gogh than short dabs, and the shader tapers their ends gracefully
    max_steps = int(np.clip(radius * 8.0 / step, 4, 18))

    def inside(p: Tuple[float, float]) -> bool:
        # floor semantics: fractional-negative coordinates are OUT of bounds
        # ( round() would let points up to 0.5 px outside slip through )
        xi, yi = int(np.floor(p[0])), int(np.floor(p[1]))
        return 0 <= xi < w and 0 <= yi < h and mask[yi, xi]

    def walk(direction: np.ndarray, sign: float) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        d = direction * sign
        p = np.array(seed, np.float32)
        for _ in range(max_steps):
            nxt = p + d * step
            if not inside(nxt):
                break
            xi = int(np.clip(round(nxt[0]), 0, w - 1))
            yi = int(np.clip(round(nxt[1]), 0, h - 1))
            cand = _iso_direction(gx, gy, xi, yi, d)
            d = 0.35 * cand + 0.65 * d
            nd = float(np.hypot(*d))
            if nd < 1e-6:
                break
            d = d / nd
            if float(np.dot(d, direction * sign)) < np.cos(np.deg2rad(75.0)):
                break
            p = nxt
            pts.append((float(p[0]), float(p[1])))
        return pts

    forward = walk(dir0, +1.0)
    backward = walk(dir0, -1.0)
    chain = list(reversed(backward)) + [seed] + forward
    return chain


def generate_stroke_segments(image_rgb: np.ndarray, layer_masks: Dict[str, np.ndarray],
                             paint_order: Sequence[str], max_segments: int = 48000,
                             rng_seed: int = 7) -> Tuple[np.ndarray, Dict]:
    """Run the painterly pass over every layer.

    Returns ``( segments ( N, 12 ) float32, meta dict )``.
    """
    import cv2

    rng = np.random.default_rng(rng_seed)
    h, w = image_rgb.shape[:2]
    rgb01 = image_rgb.astype(np.float32) / 255.0
    lum = _luminance(rgb01)
    radius_scale = max(h, w) / REFERENCE_SIDE

    # ---- per-layer segment budgets ( proportional to layer area ) --------
    # a single global cap would let the first layer ( sky ) consume the
    # whole budget before the village / moon ever get painted.
    areas = {name: float(layer_masks[name].sum()) for name in paint_order if name in layer_masks}
    total_area = sum(areas.values()) or 1.0
    budgets = {name: max(1200, int(max_segments * area / total_area))
               for name, area in areas.items()}

    segments: List[List[float]] = []
    layer_counts: Dict[str, int] = {}
    for layer_idx, name in enumerate(paint_order):
        mask = layer_masks.get(name)
        if mask is None or not mask.any():
            continue
        scales = [r * radius_scale for r in LAYER_SCALES.get(name, DEFAULT_SCALES)]
        if len(paint_order) > 6:  # generic mode: uniform stroke scale
            scales = [r * radius_scale for r in DEFAULT_SCALES]
        budget = budgets[name]
        start_count = len(segments)

        for radius in scales:
            if len(segments) - start_count >= budget:
                break
            sigma = max(radius * 0.45, 0.8)
            lum_blur = cv2.GaussianBlur(lum, (0, 0), sigma)
            color_blur = cv2.GaussianBlur(rgb01, (0, 0), max(sigma * 0.8, 0.6))
            gx, gy = _field_gradient(lum_blur)

            spacing = max(radius * 1.35, 2.0)   # denser seed grid -> lusher paint
            nx = int(np.ceil(w / spacing))
            ny = int(np.ceil(h / spacing))
            jitter_x = rng.uniform(0, spacing, size=(ny + 1, nx + 1))
            jitter_y = rng.uniform(0, spacing, size=(ny + 1, nx + 1))

            for jy in range(ny + 1):
                for jx in range(nx + 1):
                    if len(segments) - start_count >= budget:
                        break
                    seed_x = jx * spacing + jitter_x[jy, jx]
                    seed_y = jy * spacing + jitter_y[jy, jx]
                    xi, yi = int(seed_x), int(seed_y)
                    if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
                        continue

                    dir0 = _iso_direction(gx, gy, xi, yi,
                                          np.array([np.cos(rng.uniform(0, 2 * np.pi)),
                                                    np.sin(rng.uniform(0, 2 * np.pi))], np.float32))
                    chain = _grow_chain((seed_x, seed_y), dir0, mask, gx, gy, radius)
                    if len(chain) < 2:
                        continue

                    base_opacity = float(rng.uniform(0.78, 0.92))
                    phase = float(rng.uniform(0, 2 * np.pi))
                    color_jitter = float(rng.uniform(-0.04, 0.04))
                    # per-stroke thickness jitter: no two dabs are identical,
                    # the paint surface gains organic variety
                    radius_eff = float(radius) * (0.88 + 0.24 * float(rng.random()))
                    n_links = len(chain) - 1

                    for k, (p0, p1) in enumerate(zip(chain[:-1], chain[1:])):
                        cx = (p0[0] + p1[0]) * 0.5
                        cy = (p0[1] + p1[1]) * 0.5
                        cx = float(np.clip(cx, 0.0, w - 1.0))   # keep the bin clean
                        cy = float(np.clip(cy, 0.0, h - 1.0))
                        dvec = np.array([p1[0] - p0[0], p1[1] - p0[1]], np.float32)
                        dlen = float(np.hypot(*dvec))
                        if dlen < 0.35:
                            continue
                        dvec /= dlen
                        cxi = int(np.clip(round(cx), 0, w - 1))
                        cyi = int(np.clip(round(cy), 0, h - 1))
                        color = color_blur[cyi, cxi]
                        color = np.clip(color * (1.0 + color_jitter), 0.0, 1.0)
                        seg_len = dlen + radius_eff * 0.55  # overlap for continuity
                        # opacity envelope along the chain: strokes land softly
                        # and lift off at the ends instead of stopping dead
                        env = 0.62 + 0.38 * math.sin(math.pi * (k + 0.5) / max(n_links, 1))
                        segments.append([
                            float(cx), float(cy), float(dvec[0]), float(dvec[1]),
                            float(seg_len), radius_eff, base_opacity * env, float(layer_idx),
                            float(color[0]), float(color[1]), float(color[2]),
                            phase,
                        ])

        layer_counts[name] = len(segments) - start_count

    array = np.asarray(segments, dtype=np.float32).reshape(-1, SEGMENT_STRIDE) if segments \
        else np.zeros((0, SEGMENT_STRIDE), np.float32)
    meta = {
        "count": int(array.shape[0]),
        "strideFloats": SEGMENT_STRIDE,
        "layers": list(paint_order),
        "segmentsPerLayer": layer_counts,
        "radiusRange": [float(array[:, 5].min()), float(array[:, 5].max())] if len(array) else [0, 0],
        "lengthRange": [float(array[:, 4].min()), float(array[:, 4].max())] if len(array) else [0, 0],
        "referenceSide": REFERENCE_SIDE,
    }
    LOGGER.info("Painterly pass: %d stroke segments across %d layers",
                meta["count"], len(paint_order))
    return array, meta


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def _load_layer_masks(out_dir: Path, manifest: List[dict]) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for entry in manifest:
        path = out_dir / entry["file"]
        if not path.exists():
            continue
        alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
        masks[entry["name"]] = alpha > 127
    return masks


def run_strokes(image, out_dir: Path | str, max_segments: int = 48000,
                force: bool = False) -> dict:
    """Generate brush strokes for ``image`` using the SAM layers.

    Writes ``<out_dir>/strokes/strokes.bin`` + ``strokes.json`` and returns
    the manifest fragment for the web export.
    """
    if isinstance(image, (str, Path)):
        image_rgb = utils.load_image_rgb(image)
    else:
        image_rgb = np.asarray(image)

    out_dir = ensure_dir(Path(out_dir))
    strokes_dir = ensure_dir(out_dir / "strokes")
    json_path = strokes_dir / "strokes.json"
    if json_path.exists() and not force:
        LOGGER.info("strokes.json exists - skipping stroke stage ( use --force to rerun )")
        return utils.read_json(json_path)

    layers_json = out_dir / "layers.json"
    if not layers_json.exists():
        raise FileNotFoundError("layers.json missing - run the SAM stage first")
    layers_manifest: List[dict] = utils.read_json(layers_json)
    masks = _load_layer_masks(out_dir, layers_manifest)
    if not masks:
        raise RuntimeError("No layer masks could be loaded")

    with utils.timed("Hertzmann painterly stroke generation"):
        segments, meta = generate_stroke_segments(
            image_rgb, masks, [e["name"] for e in layers_manifest],
            max_segments=max_segments)

    with utils.timed("Stroke serialisation"):
        segments.tofile(strokes_dir / "strokes.bin")
        meta.update({
            "bin": "strokes/strokes.bin",
            "json": "strokes/strokes.json",
            "imageWidth": int(image_rgb.shape[1]),
            "imageHeight": int(image_rgb.shape[0]),
        })
        atomic_write_json(json_path, meta)

    LOGGER.info("Strokes written to %s", strokes_dir)
    return meta
