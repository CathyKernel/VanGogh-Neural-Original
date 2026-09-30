"""Stage 1 - SAM semantic decomposition into painter layers.

Runs Meta AI's Segment Anything Model (SAM) in *automatic mask generation*
mode, then groups the resulting class-agnostic masks into the semantic
layers of *The Starry Night* composition using classical CV cues
(HSV statistics, position priors, size and shape descriptors)::

    layers/sky.png, layers/mountains.png, layers/village.png,
    layers/tree.png, layers/stars.png, layers/moon.png

Each layer is an RGBA PNG: RGB copied verbatim from the original painting
(colour preservation) with alpha = the layer mask, so the WebGL renderer can
composite layers back-to-front in paint order.

Robustness
----------
* ``model_type='auto'`` picks the best checkpoint that exists on disk
  (ViT-H > ViT-B > ViT-L).
* If no checkpoint / no ``segment_anything`` package is available the stage
  falls back to a classical pipeline (Lab k-means colour quantisation +
  connected components) so the demo always produces layers.
* Bright yellow blobs (stars / moon) that SAM occasionally misses are
  recovered with an HSV + connected-components blob detector.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import utils
from .utils import ensure_dir, get_logger, save_rgba_png, atomic_write_json

LOGGER = get_logger()

# paint order = back-to-front ( sky painted first, moon on top )
LAYER_PAINT_ORDER: Dict[str, int] = {
    "sky": 0,
    "mountains": 1,
    "village": 2,
    "tree": 3,
    "stars": 4,
    "moon": 5,
}

MODEL_FILES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}

MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

_AMG_CACHE: Dict[Tuple[str, str], object] = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LayerSpec:
    """A semantic layer ready for rendering / export."""

    name: str
    paint_order: int
    mask: np.ndarray  # bool (H, W) - union of all member masks
    mean_color: Tuple[int, int, int]

    @property
    def area_frac(self) -> float:
        return float(self.mask.mean())


# ---------------------------------------------------------------------------
# SAM model handling
# ---------------------------------------------------------------------------

def resolve_model_type(model_type: str = "auto", ckpt_dir: Path | str | None = None) -> Optional[str]:
    """Return the SAM variant to use, or None when no checkpoint exists."""
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else utils.repo_root() / "checkpoints"
    if model_type and model_type != "auto":
        return model_type if (ckpt_dir / MODEL_FILES[model_type]).exists() else None
    for candidate in ("vit_h", "vit_l", "vit_b"):
        if (ckpt_dir / MODEL_FILES[candidate]).exists():
            return candidate
    return None


def load_amg(model_type: str, ckpt_dir: Path | str | None = None,
             device: str = "auto", points_per_side: Optional[int] = None):
    """Build (and cache) a SamAutomaticMaskGenerator for the given variant."""
    key = (model_type, str(device))
    if key in _AMG_CACHE:
        return _AMG_CACHE[key]

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else utils.repo_root() / "checkpoints"
    ckpt_path = ckpt_dir / MODEL_FILES[model_type]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")

    dev = utils.torch_device(device)
    LOGGER.info("Loading SAM %s from %s on %s", model_type, ckpt_path.name, dev)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(device=dev)
    sam.eval()

    if points_per_side is None:
        points_per_side = 32 if dev.type == "cuda" else 16

    amg = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        crop_n_layers=0,
        min_mask_region_area=int(0.0002 * (512 * 512)),  # denoise micro specks
    )
    _AMG_CACHE.clear()  # only one SAM resident at a time ( RAM friendly )
    _AMG_CACHE[key] = amg
    return amg


def release_models() -> None:
    """Free cached SAM models (called between pipeline stages)."""
    _AMG_CACHE.clear()


def generate_sam_masks(image_rgb: np.ndarray, amg) -> List[dict]:
    """Run automatic mask generation; returns masks sorted by area (desc)."""
    import torch

    with torch.inference_mode():
        raw = amg.generate(image_rgb)
    masks = [m for m in raw if m["area"] >= 64]
    masks.sort(key=lambda m: m["area"], reverse=True)
    LOGGER.info("SAM produced %d usable masks", len(masks))
    return masks


# ---------------------------------------------------------------------------
# Classical fallback segmentation (k-means colour quantisation)
# ---------------------------------------------------------------------------

def classical_masks(image_rgb: np.ndarray, k: int = 7) -> List[dict]:
    """No-SAM fallback: Lab colour quantisation + connected components.

    Produces SAM-like mask dicts ( ``segmentation`` / ``area`` / ``bbox`` )
    so downstream labelling code is model agnostic.
    """
    import cv2

    h, w = image_rgb.shape[:2]
    small = cv2.resize(image_rgb, (256, max(1, int(256 * h / w))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.8)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])

    masks: List[dict] = []
    for cluster in range(k):
        comp = (labels == cluster).astype(np.uint8)
        comp = cv2.resize(comp, (w, h), interpolation=cv2.INTER_NEAREST)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=8)
        for cid in range(1, n):
            seg = cc[cid].astype(bool)
            area = int(seg.sum())
            if area < 200:
                continue
            x, y, bw, bh, _ = stats[cid]
            masks.append({
                "segmentation": seg,
                "area": area,
                "bbox": [int(x), int(y), int(bw), int(bh)],
            })
    masks.sort(key=lambda m: m["area"], reverse=True)
    LOGGER.info("Classical fallback segmentation produced %d masks", len(masks))
    return masks


# ---------------------------------------------------------------------------
# Semantic labelling - Starry Night priors
# ---------------------------------------------------------------------------

def _mask_stats(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Geometry + colour statistics used by the labelling rules."""
    from .utils import rgb_to_hsv

    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    hsv = rgb_to_hsv(image_rgb[mask].astype(np.float32) / 255.0)
    hue = float(np.mean(hsv[:, 0]))
    return {
        "area_frac": float(mask.mean()),
        "cx": float(xs.mean() / w),
        "cy": float(ys.mean() / h),
        "bw": (x1 - x0 + 1) / w,
        "bh": (y1 - y0 + 1) / h,
        "fill": float(mask.sum() / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))),
        "aspect": float((y1 - y0 + 1) / max(1, x1 - x0 + 1)),
        "hue": hue,  # 0..1 fraction of the hue circle
        "sat": float(np.mean(hsv[:, 1])),
        "val": float(np.mean(hsv[:, 2])),
        "val_std": float(np.std(hsv[:, 2])),
    }


def is_yellowish(s: dict) -> bool:
    """Van Gogh star / moon yellows sit around hue 0.08-0.25 (OpenCV scale)."""
    return 0.06 <= s["hue"] <= 0.30 and s["sat"] >= 0.25


def classify_mask(s: dict) -> str:
    """Assign one SAM mask a semantic layer name from its statistics."""
    a = s["area_frac"]
    if 0.0015 <= a <= 0.03 and s["val"] > 0.55 and is_yellowish(s) \
            and s["cx"] > 0.45 and s["cy"] < 0.55 and s["fill"] > 0.45:
        return "moon"
    if a < 0.006 and s["val"] > 0.5 and is_yellowish(s) and s["cy"] < 0.62:
        return "stars"
    if s["cy"] > 0.70 and a > 0.0008:
        return "village"
    if (s["cx"] < 0.24 and s["bh"] > 0.22) or \
            (s["aspect"] > 1.7 and s["val"] < 0.4 and a > 0.004) or \
            (s["val"] < 0.30 and a > 0.015 and s["cx"] < 0.38):
        return "tree"
    if 0.30 <= s["cy"] <= 0.72 and s["val"] < 0.5 and not is_yellowish(s) and a > 0.01:
        return "mountains"
    return "sky"


def label_sam_masks(image_rgb: np.ndarray, masks: Sequence[dict],
                    labels_mode: str = "auto") -> Dict[str, np.ndarray]:
    """Group all masks into named layer masks ( union per label )."""
    h, w = image_rgb.shape[:2]
    layers: Dict[str, np.ndarray] = {}

    if labels_mode == "generic":
        # depth-proxy ordering: lower centroid first ( far away, painted first )
        order = {
            "layer_0": 0, "layer_1": 1, "layer_2": 2,
            "layer_3": 3, "layer_4": 4, "layer_5": 5,
        }
        LAYER_PAINT_ORDER.update(order)
        ranked = sorted(masks, key=lambda m: _mask_stats(image_rgb, m["segmentation"])["cy"])
        for i, m in enumerate(ranked[:6]):
            name = f"layer_{i}"
            layers[name] = layers.get(name, np.zeros((h, w), bool)) | m["segmentation"]
        layers.setdefault("layer_0", np.ones((h, w), bool))
        return layers

    for m in masks:
        stats = _mask_stats(image_rgb, m["segmentation"])
        label = classify_mask(stats)
        if label == "sky" and stats["area_frac"] < 0.02 and stats["val_std"] < 0.12:
            continue  # boring tiny fragment, drop it
        layers[label] = layers.get(label, np.zeros((h, w), bool)) | m["segmentation"]

    # sky is the painted base: covers anything the other layers overwrite
    layers.setdefault("sky", np.zeros((h, w), bool))
    layers["sky"] |= ~_union(layers, exclude="sky")
    return layers


def _union(layers: Dict[str, np.ndarray], exclude: str) -> np.ndarray:
    out = None
    for name, mask in layers.items():
        if name == exclude:
            continue
        out = mask.copy() if out is None else (out | mask)
    return out if out is not None else np.zeros_like(next(iter(layers.values())))


def blob_assist(image_rgb: np.ndarray, layers: Dict[str, np.ndarray],
                min_star_area: int = 12) -> None:
    """Recover missed stars / moon with HSV blob detection (in place)."""
    import cv2
    from .utils import rgb_to_hsv

    h, w = image_rgb.shape[:2]
    hsv = rgb_to_hsv(image_rgb.astype(np.float32) / 255.0)
    yellow = (hsv[..., 0] >= 0.06) & (hsv[..., 0] <= 0.30) & \
             (hsv[..., 1] >= 0.22) & (hsv[..., 0] >= 0) & (hsv[..., 2] >= 0.5)
    yellow_u8 = (yellow * 255).astype(np.uint8)
    yellow_u8 = cv2.morphologyEx(yellow_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(yellow_u8, connectivity=8)

    star_mask = layers.setdefault("stars", np.zeros((h, w), bool))
    moon_mask = layers.setdefault("moon", np.zeros((h, w), bool))
    total = h * w

    best_moon: Optional[int] = None
    best_moon_area = 0
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        cx = stats[cid, cv2.CC_STAT_LEFT] + stats[cid, cv2.CC_STAT_WIDTH] / 2.0
        cy = stats[cid, cv2.CC_STAT_TOP] + stats[cid, cv2.CC_STAT_HEIGHT] / 2.0
        if area < min_star_area:
            continue
        if area <= 0.004 * total:
            star_mask |= cc == cid
        elif 0.004 * total < area <= 0.05 * total and cy < 0.55 * h:
            if area > best_moon_area:
                best_moon, best_moon_area = cid, area

    if best_moon is not None and not moon_mask.any():
        # moon detected - keep stars from spilling into its halo
        moon_mask |= cc == best_moon
        star_mask &= ~moon_mask
        LOGGER.info("Blob assist recovered the moon ( %d px )", best_moon_area)
    recovered = int(star_mask.sum())
    if recovered:
        LOGGER.info("Blob assist recovered %d star pixels", recovered)


# ---------------------------------------------------------------------------
# Layer stack assembly
# ---------------------------------------------------------------------------

def build_layer_stack(image_rgb: np.ndarray, layers: Dict[str, np.ndarray]) -> List[LayerSpec]:
    """Sort layers in paint order, attach mean colours, drop empties."""
    specs: List[LayerSpec] = []
    for name, mask in layers.items():
        if not mask.any() and name != "sky":
            continue
        mask = mask if name == "sky" else mask & True
        pixels = image_rgb[mask] if mask.any() else np.zeros((1, 3), np.uint8)
        mean_color = tuple(int(round(c)) for c in pixels.mean(axis=0))
        specs.append(LayerSpec(name=name, paint_order=LAYER_PAINT_ORDER.get(name, 0),
                               mask=mask, mean_color=mean_color))
    specs.sort(key=lambda s: (s.paint_order, s.name))
    # keep at most 6 layers ( matches the WebGL sampler budget )
    specs = specs[:6]
    for i, spec in enumerate(specs):
        spec.paint_order = i
    return specs


def layer_specs_to_json(specs: Sequence[LayerSpec], image_shape) -> List[dict]:
    h, w = image_shape[:2]
    out = []
    for spec in specs:
        ys, xs = np.nonzero(spec.mask)
        bbox = [float(xs.min() / w), float(ys.min() / h),
                float((xs.max() + 1) / w), float((ys.max() + 1) / h)] if len(ys) else [0, 0, 1, 1]
        out.append({
            "name": spec.name,
            "paintOrder": spec.paint_order,
            "file": f"layers/{spec.name}.png",
            "areaFrac": round(spec.area_frac, 5),
            "bbox": [round(v, 4) for v in bbox],
            "meanColor": list(spec.mean_color),
        })
    return out


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_sam(image, out_dir: Path | str, model_type: str = "auto",
            ckpt_dir: Path | str | None = None, device: str = "auto",
            points_per_side: Optional[int] = None, labels_mode: str = "auto",
            force: bool = False) -> List[dict]:
    """Segment ``image`` into semantic layers and write them under ``out_dir``.

    Parameters
    ----------
    image:
        Path to the input painting or an uint8 RGB array.
    out_dir:
        Scene output directory ( ``layers/`` and ``layers.json`` are written
        inside it ).
    model_type:
        ``vit_h`` / ``vit_l`` / ``vit_b`` / ``auto`` / ``classical``.
    device:
        ``auto`` / ``cuda`` / ``cpu`` / ``mps``.
    points_per_side:
        SAM point-grid density ( auto: 32 on CUDA, 16 on CPU ).
    labels_mode:
        ``auto`` (Starry-Night priors) or ``generic`` (positional layering).

    Returns
    -------
    The layer manifest entries ( also persisted to ``layers.json`` ).
    """
    if isinstance(image, (str, Path)):
        image_rgb = utils.load_image_rgb(image)
    else:
        image_rgb = np.asarray(image)

    out_dir = ensure_dir(Path(out_dir))
    layers_dir = ensure_dir(out_dir / "layers")
    json_path = out_dir / "layers.json"
    if json_path.exists() and not force:
        LOGGER.info("layers.json exists - skipping SAM stage ( use --force to rerun )")
        return utils.read_json(json_path)

    variant = None
    if model_type not in ("classical",):
        variant = resolve_model_type(model_type, ckpt_dir)

    if variant is not None:
        amg = load_amg(variant, ckpt_dir, device, points_per_side)
        with utils.timed(f"SAM automatic mask generation ({variant})"):
            masks = generate_sam_masks(image_rgb, amg)
        release_models()
        engine = f"sam_{variant}"
    else:
        LOGGER.warning("No SAM checkpoint found - using classical k-means fallback")
        with utils.timed("Classical k-means segmentation"):
            masks = classical_masks(image_rgb)
        engine = "classical_kmeans"

    with utils.timed("Semantic layer grouping"):
        layers = label_sam_masks(image_rgb, masks, labels_mode=labels_mode)
        if labels_mode == "auto":
            blob_assist(image_rgb, layers)
        specs = build_layer_stack(image_rgb, layers)

    for spec in specs:
        save_rgba_png(layers_dir / f"{spec.name}.png", image_rgb, spec.mask)

    manifest = layer_specs_to_json(specs, image_rgb.shape)
    for entry in manifest:
        entry["engine"] = engine
    atomic_write_json(json_path, manifest)
    LOGGER.info("Wrote %d layers to %s ( engine: %s )", len(manifest), layers_dir, engine)
    return manifest
