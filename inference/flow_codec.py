"""Optical flow encoding / decoding / visualisation utilities.

Supports three representations of a dense flow field ``(u, v)``:

* ``.flo``   - the classic Middlebury binary layout ( interoperable with
               the RAFT / KITTI / Middlebury evaluation tooling ).
* 8-bit PNG  - a *decodable* RGB encoding used by the WebGL renderer
               ( R -> u, G -> v, quantised around a signed range +-scale ).
* color wheel- a qualitative visualisation PNG following the Middlebury
               colour convention ( hue = direction, value = magnitude ).

The PNG codec is intentionally invertible: the renderer reconstructs the
flow vector field in GLSL with an identical formula, so encoder and decoder
must stay in sync ( see ``renderer/shaders/optical_flow.frag`` ).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .utils import get_logger

LOGGER = get_logger()

_MIDDLEBURY_TAG = 202021.25  # the standard Middlebury float32-representable tag


# ---------------------------------------------------------------------------
# Middlebury .flo
# ---------------------------------------------------------------------------

def write_flo(path: Path | str, flow: np.ndarray) -> Path:
    """Write a (H, W, 2) float32 flow field in Middlebury format."""
    flow = np.asarray(flow, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"expected (H, W, 2), got {flow.shape}"
    h, w = flow.shape[:2]
    with open(path, "wb") as fh:
        np.array([_MIDDLEBURY_TAG, w, h], dtype=np.float32).tofile(fh)
        flow.tofile(fh)
    return Path(path)


def read_flo(path: Path | str) -> np.ndarray:
    """Read a Middlebury .flo file into (H, W, 2) float32."""
    with open(path, "rb") as fh:
        tag, w, h = np.fromfile(fh, dtype=np.float32, count=3)
        if float(tag) != _MIDDLEBURY_TAG:
            raise ValueError(f"{path}: bad Middlebury tag {tag}")
        flow = np.fromfile(fh, dtype=np.float32).reshape(int(h), int(w), 2)
    return flow


# ---------------------------------------------------------------------------
# 8-bit PNG codec (renderer facing)
# ---------------------------------------------------------------------------

def encode_flow_png(flow: np.ndarray) -> Tuple[np.ndarray, float]:
    """Encode flow into an 8-bit RGB image plus the signed scale factor.

    Encoding (must mirror ``optical_flow.frag``)::

        R = round((u / scale + 1) * 127.5)
        G = round((v / scale + 1) * 127.5)
        B = round(clamp(|flow| / scale, 0, 1) * 255)   # magnitude hint

    ``scale`` is the 99th percentile of the flow magnitude (robust to
    outliers), floored to 1e-4.  Quantisation error is bounded by
    ``scale / 255`` per axis which is sub-pixel for typical scales.

    Returns ``(rgb_uint8, scale)``.
    """
    flow = np.asarray(flow, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"expected (H, W, 2), got {flow.shape}"
    mag = np.linalg.norm(flow, axis=2)
    scale = max(float(np.percentile(mag, 99.0)), 1e-4)
    u, v = flow[..., 0], flow[..., 1]
    r = np.clip(np.round((u / scale + 1.0) * 127.5), 0, 255).astype(np.uint8)
    g = np.clip(np.round((v / scale + 1.0) * 127.5), 0, 255).astype(np.uint8)
    b = np.clip(mag / scale, 0.0, 1.0)
    b = (b * 255.0 + 0.5).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    return rgb, float(scale)


def decode_flow_png(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Decode an 8-bit RGB flow PNG back into (H, W, 2) float32."""
    rgb = np.asarray(rgb, dtype=np.float32)
    u = (rgb[..., 0] / 127.5 - 1.0) * scale
    v = (rgb[..., 1] / 127.5 - 1.0) * scale
    return np.dstack([u, v]).astype(np.float32)


def save_flow_png(path: Path | str, flow: np.ndarray) -> Tuple[Path, float]:
    """Encode + save; returns (path, scale) so callers can persist the scale."""
    rgb, scale = encode_flow_png(flow)
    Image.fromarray(rgb).save(path)
    return Path(path), scale


def load_flow_png(path: Path | str, scale: float) -> np.ndarray:
    return decode_flow_png(np.asarray(Image.open(path).convert("RGB")), scale)


# ---------------------------------------------------------------------------
# Middlebury colour wheel visualisation
# ---------------------------------------------------------------------------

def _make_colorwheel() -> np.ndarray:
    """Classic Middlebury 55-entry colour wheel (RGB float rows)."""
    rays = 15
    wheel = []
    # red -> yellow (hue 0 -> 60)
    wheel += [(1.0, i / rays, 0.0) for i in range(rays)]
    # yellow -> green
    wheel += [(1.0 - i / rays, 1.0, 0.0) for i in range(rays)]
    # green -> cyan (skip pure duplicates)
    wheel += [(0.0, 1.0, i / rays) for i in range(rays)]
    # cyan -> blue
    wheel += [(0.0, 1.0 - i / rays, 1.0) for i in range(rays)]
    # blue -> magenta
    wheel += [(i / rays, 0.0, 1.0) for i in range(rays)]
    # magenta -> red
    wheel += [(1.0, 0.0, 1.0 - i / rays) for i in range(rays)]
    return np.asarray(wheel, dtype=np.float32)


_COLORWHEEL = _make_colorwheel()


def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Render a flow field using the Middlebury colour convention.

    Returns an uint8 RGB array (H, W, 3) suitable for quick human inspection.
    """
    flow = np.asarray(flow, dtype=np.float32)
    h, w = flow.shape[:2]
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99.0)) or 1e-4
    max_mag = max(max_mag, 1e-4)
    mag = np.clip(mag / max_mag, 0.0, 1.0)

    angle = np.arctan2(-v, -u) / np.pi  # in [-1, 1]
    frac = (angle + 1.0) * 0.5 * (_COLORWHEEL.shape[0] - 1e-6)
    idx = np.floor(frac).astype(np.int32)
    frac_next = frac - idx
    next_idx = (idx + 1) % _COLORWHEEL.shape[0]

    color = (1.0 - frac_next[..., None]) * _COLORWHEEL[idx] + frac_next[..., None] * _COLORWHEEL[next_idx]
    hsv_v = mag[..., None]
    rgb = color * hsv_v + (1.0 - hsv_v) * 0.65  # faint grey background where flow ~ 0
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
