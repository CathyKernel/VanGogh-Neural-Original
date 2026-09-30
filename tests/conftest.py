"""Shared pytest fixtures.  No model downloads: everything runs on
synthetic fixtures so the suite is CI-friendly and finishes in seconds."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def painting() -> np.ndarray:
    """A tiny synthetic Starry-Night-like painting ( 96 x 128 RGB ).

    Layout mirrors the real composition so the starry-night heuristics have
    something honest to bite on.
    """
    h, w = 96, 128
    img = np.zeros((h, w, 3), np.uint8)
    # deep blue sky
    img[:, :] = (30, 60, 120)
    # moon: bright yellow blob, upper right
    yy, xx = np.mgrid[0:h, 0:w]
    moon = ((xx - 100) ** 2 + (yy - 18) ** 2) < 8 ** 2
    img[moon] = (250, 240, 160)
    # stars: small yellow dots in the upper half
    for sx, sy in [(20, 10), (40, 28), (60, 12), (80, 34), (14, 40)]:
        star = ((xx - sx) ** 2 + (yy - sy) ** 2) < 3 ** 2
        img[star] = (255, 230, 120)
    # mountains: dark band
    img[60:72, :] = (40, 35, 60)
    # village: bottom band with lit windows
    img[72:, :] = (35, 40, 55)
    img[80:88, 20:26] = (240, 200, 100)
    # cypress: dark tall left shape
    cypress = (xx < 12) & (yy > 20)
    img[cypress] = (25, 35, 28)
    return img


@pytest.fixture()
def flat_depth(painting: np.ndarray) -> np.ndarray:
    """Constant mid-depth map sized like the painting fixture."""
    return np.full(painting.shape[:2], 0.5, np.float32)
