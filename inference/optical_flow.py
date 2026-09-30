"""Stage 3 - RAFT dense optical flow on a depth-synthesised parallax pair.

A painting is a single still image, so there is no second video frame to
compute flow from.  Following the painterly-animation literature
( Hays & Essa 2007; O'Donovan & Hertzmann 2014 ) we *synthesise* the second
frame ourselves: the estimated depth map drives a synthetic camera
translation, producing a parallax pair.  RAFT ( Teed & Deng, ECCV 2020 -
official torchvision implementation with pretrained weights ) then computes
the dense flow field between the original and the parallax frame.

The resulting flow is exactly the motion the WebGL renderer later applies
to brush strokes: swirl-heavy sky regions receive large coherent motion,
near-field layers ( cypress, moon ) move with stronger parallax.

Outputs ( under ``<scene>/flow`` )
----------------------------------
``flow.npy``    float32 ( H, W, 2 ), pixels
``flow.flo``    Middlebury format for standard flow tooling
``flow.png``    8-bit invertible RGB codec consumed by WebGL
``flow_viz.png``Middlebury colour-wheel visualisation
``flow.json``   manifest fragment ( scale, engine, stats )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from . import utils
from .flow_codec import flow_to_color, save_flow_png, write_flo
from .utils import ensure_dir, get_logger, atomic_write_json

LOGGER = get_logger()

_RAFT_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_raft(prefer: str = "large", device: str = "auto"):
    """Load torchvision RAFT with pretrained weights.

    Priority: ``raft_large`` -> ``raft_small`` -> ``None`` ( caller then uses
    the analytic parallax fallback ).
    Returns ``(model, transform, engine_name)``.
    """
    import torch
    from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

    dev = utils.torch_device(device)
    attempts = [("large", raft_large, Raft_Large_Weights),
                ("small", raft_small, Raft_Small_Weights)]
    if prefer == "small":
        attempts.reverse()

    for size, builder, weights_cls in attempts:
        cache_key = f"raft_{size}_{dev.type}"
        if cache_key in _RAFT_CACHE:
            return (*_RAFT_CACHE[cache_key], f"raft_{size}")
        try:
            weights = weights_cls.DEFAULT
            model = builder(weights=weights).to(dev).eval()
            bundle = (model, weights.transforms())
            _RAFT_CACHE.clear()
            _RAFT_CACHE[cache_key] = bundle
            LOGGER.info("Loaded torchvision RAFT-%s with pretrained weights", size)
            return model, weights.transforms(), f"raft_{size}"
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("RAFT-%s weights unavailable ( %s )", size, exc)
    return None, None, None


def release_models() -> None:
    _RAFT_CACHE.clear()


# ---------------------------------------------------------------------------
# Parallax pair synthesis
# ---------------------------------------------------------------------------

def synthesize_parallax_pair(image_rgb: np.ndarray, depth01: np.ndarray,
                             shift_xy: Tuple[float, float]) -> np.ndarray:
    """Render the second frame of a synthetic camera translation.

    ``frame2(x) = frame1(x - shift * depth(x))`` via inverse bilinear warp
    ( ``torch.nn.functional.grid_sample`` ), i.e. near pixels displace more -
    the geometry of a real camera pan.  Returns the warped frame ( uint8 ).
    """
    import torch
    import torch.nn.functional as F

    h, w = image_rgb.shape[:2]
    frame = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()[None] / 255.0
    depth = torch.from_numpy(np.ascontiguousarray(depth01, dtype=np.float32))

    dx, dy = shift_xy
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    src_x = xs - dx * depth
    src_y = ys - dy * depth
    grid = torch.stack([src_x / max(w - 1, 1) * 2 - 1,
                        src_y / max(h - 1, 1) * 2 - 1], dim=-1)[None]
    warped = F.grid_sample(frame, grid, mode="bilinear",
                           padding_mode="reflection", align_corners=True)
    out = (warped[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# RAFT inference
# ---------------------------------------------------------------------------

def _raft_flow(model, transform, image1: np.ndarray, image2: np.ndarray,
               device: str = "auto") -> np.ndarray:
    """Run RAFT and resample its output back to the input resolution.

    torchvision RAFT test-time transforms resize inputs to a fixed
    ``(520, 960)`` canvas; flow vectors are rescaled back to native pixel
    units afterwards.
    """
    import torch
    import cv2

    dev = utils.torch_device(device)

    def to_tensor(img: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255.0

    with torch.inference_mode():
        x1, x2 = transform(to_tensor(image1), to_tensor(image2))
        flows = model(x1.to(dev), x2.to(dev))
    flow_low = flows[-1][0].cpu().numpy()  # ( 2, 520, 960 )
    _, fh, fw = flow_low.shape
    h, w = image1.shape[:2]

    u = cv2.resize(flow_low[0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / fw)
    v = cv2.resize(flow_low[1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / fh)
    return np.dstack([u, v]).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_raft(image, out_dir: Path | str, device: str = "auto", prefer: str = "large",
             shift_fraction: Tuple[float, float] = (0.04, 0.015),
             force: bool = False) -> dict:
    """Compute the depth-parallax optical flow for ``image``.

    Reads ``<out_dir>/depth/depth_smooth.npy`` ( stage 2 ) and writes the
    flow assets under ``<out_dir>/flow``.  Returns the manifest fragment.
    """
    if isinstance(image, (str, Path)):
        image_rgb = utils.load_image_rgb(image)
    else:
        image_rgb = np.asarray(image)

    out_dir = ensure_dir(Path(out_dir))
    flow_dir = ensure_dir(out_dir / "flow")
    json_path = flow_dir / "flow.json"
    if json_path.exists() and not force:
        LOGGER.info("flow.json exists - skipping flow stage ( use --force to rerun )")
        return utils.read_json(json_path)

    h, w = image_rgb.shape[:2]
    depth_path = out_dir / "depth" / "depth_smooth.npy"
    if depth_path.exists():
        depth = np.load(depth_path)
    else:
        LOGGER.warning("No depth map found - using flat depth ( parallax disabled )")
        depth = np.full((h, w), 0.5, np.float32)

    shift_xy = (shift_fraction[0] * w, shift_fraction[1] * h)

    with utils.timed("Parallax pair synthesis"):
        frame2 = synthesize_parallax_pair(image_rgb, depth, shift_xy)
        Image.fromarray(frame2).save(flow_dir / "parallax_pair.png")

    model, transform, engine = load_raft(prefer, device)
    if model is not None:
        with utils.timed(f"RAFT inference ({engine})"):
            flow = _raft_flow(model, transform, image_rgb, frame2, device)
        release_models()
    else:
        LOGGER.warning("RAFT weights unavailable - analytic depth-parallax fallback")
        engine = "analytic_parallax"
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        flow = np.dstack([np.broadcast_to(-shift_xy[0], (h, w)) * depth,
                          np.broadcast_to(-shift_xy[1], (h, w)) * depth]).astype(np.float32)

    with utils.timed("Flow export"):
        np.save(flow_dir / "flow.npy", flow)
        write_flo(flow_dir / "flow.flo", flow)
        flow_path, scale = save_flow_png(flow_dir / "flow.png", flow)
        Image.fromarray(flow_to_color(flow)).save(flow_dir / "flow_viz.png")

        mag = np.linalg.norm(flow, axis=2)
        manifest = {
            "engine": engine,
            "file": "flow/flow.png",
            "rawFile": "flow/flow.npy",
            "floFile": "flow/flow.flo",
            "vizFile": "flow/flow_viz.png",
            "pairFile": "flow/parallax_pair.png",
            "scale": round(scale, 4),
            "cameraShift": [round(s, 2) for s in shift_xy],
            "shiftFraction": list(shift_fraction),
            "stats": {
                "maxMag": round(float(mag.max()), 3),
                "meanMag": round(float(mag.mean()), 3),
                "p99Mag": round(float(np.percentile(mag, 99)), 3),
            },
        }
        atomic_write_json(json_path, manifest)
    LOGGER.info("Flow written to %s ( engine: %s, scale %.2f px )", flow_dir, engine, scale)
    return manifest
