"""Shared utilities: logging, image IO, JSON, device selection and timing."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

LOGGER_NAME = "vangogh"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the package logger (creates it on first use)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    get_logger().setLevel(level)


def repo_root() -> Path:
    """Absolute path of the project repository root."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create *path* (and parents) if missing and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------

def load_image_rgb(path: os.PathLike | str, max_side: Optional[int] = None) -> np.ndarray:
    """Load an image as an uint8 RGB array (H, W, 3).

    Parameters
    ----------
    path:
        Input image path (PNG/JPG, any mode; alpha is composited over black).
    max_side:
        If given, the image is resized so that its longest side equals
        ``max_side`` while preserving aspect ratio (LANCZOS resampling).
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))
    img = img.convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = float(max_side) / float(max(w, h))
        if scale < 1.0 - 1e-6:
            img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                             resample=Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def save_rgb_png(path: os.PathLike | str, rgb: np.ndarray) -> Path:
    """Save an uint8 RGB array as PNG."""
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)
    return Path(path)


def save_rgba_png(path: os.PathLike | str, rgb: np.ndarray, alpha: np.ndarray) -> Path:
    """Save an RGB array plus an alpha mask (H, W) as an RGBA PNG.

    ``alpha`` is uint8 or float in [0, 1]; fully transparent pixels keep their
    RGB value (premultiplied look is avoided in the file itself).
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be HxWx3"
    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(path)
    return Path(path)


def save_gray16_png(path: os.PathLike | str, values01: np.ndarray) -> Path:
    """Save a float map in [0, 1] as a 16-bit single channel PNG."""
    assert values01.ndim == 2, "values01 must be HxW"
    clipped = np.clip(values01, 0.0, 1.0)
    u16 = (clipped * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(u16, mode="I;16").save(path)
    return Path(path)


def load_gray16_png(path: os.PathLike | str) -> np.ndarray:
    """Read a 16-bit grayscale PNG back as float32 in [0, 1]."""
    img = Image.open(path)
    return np.asarray(img, dtype=np.float32) / 65535.0


# ---------------------------------------------------------------------------
# JSON helpers (atomic writes keep half-written manifests out of the web build)
# ---------------------------------------------------------------------------

def atomic_write_json(path: os.PathLike | str, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, p)
    return p


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def torch_device(request: str = "auto") -> "torch.device":  # noqa: F821
    """Resolve the torch device honouring the user request.

    ``auto`` prefers CUDA, then Apple MPS, then CPU.
    """
    import torch

    if request and request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: "torch.device") -> str:  # noqa: F821
    return f"{device.type}{(':' + str(device.index)) if device.index is not None else ''}"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def percentile_normalize(values: np.ndarray, lo: float = 0.5, hi: float = 99.5,
                         clip: bool = True) -> np.ndarray:
    """Robust min-max normalisation using percentiles (outlier tolerant)."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin, vmax = np.percentile(values, [lo, hi])
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    out = (values - float(vmin)) / (float(vmax) - float(vmin))
    out = out.astype(np.float32, copy=False)
    return np.clip(out, 0.0, 1.0) if clip else out


def unit01(values: np.ndarray) -> np.ndarray:
    """Plain min-max normalisation into [0, 1] (no clipping needed after)."""
    values = np.asarray(values, dtype=np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> HSV with H in [0,1) (fraction of full hue circle).

    Accepts either an image ( H, W, 3 ) or a flat pixel list ( N, 3 ); the
    output shape matches the input shape.
    """
    import cv2

    rgb01 = np.asarray(rgb01, np.float32)
    flat = rgb01.ndim == 2
    if flat:
        rgb01 = rgb01[:, None, :]
    hsv = cv2.cvtColor(np.ascontiguousarray((rgb01 * 255).astype(np.uint8)),
                       cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32) / np.array([179.0, 255.0, 255.0], dtype=np.float32)
    return hsv[:, 0, :] if flat else hsv


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timed(stage: str):
    """Context manager that logs wall-clock duration of a pipeline stage."""
    logger = get_logger()
    t0 = time.perf_counter()
    logger.info("%s ...", stage)
    yield
    logger.info("%s done in %.1fs", stage, time.perf_counter() - t0)


class StageTimer:
    """Collects per-stage durations for the run report."""

    def __init__(self) -> None:
        self._records: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        self._records[name] = self._records.get(name, 0.0) + (time.perf_counter() - t0)

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 2) for k, v in sorted(self._records.items())}


def file_sha256(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def bytes_human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"
