"""Optical flow stage: parallax pair synthesis + RAFT wrapper ( mocked )."""

import numpy as np
import pytest
from PIL import Image

from inference import optical_flow
from inference.optical_flow import _raft_flow, run_raft, synthesize_parallax_pair


def test_parallax_pair_shape_and_shift(painting, flat_depth):
    dx, dy = 8.0, 0.0
    frame2 = synthesize_parallax_pair(painting, flat_depth, (dx, dy))
    assert frame2.shape == painting.shape
    assert frame2.dtype == np.uint8

    # with constant depth the warp is a pure translation by +dx:
    # interior columns of frame2 equal columns of frame1 shifted left by dx
    src = painting.astype(np.int16)
    got = frame2.astype(np.int16)
    y = painting.shape[0] // 2
    interior_err = np.abs(got[y, 20:-20] - src[y, 20 + int(dx): -20 + int(dx)])
    assert interior_err.mean() < 2.0  # bilinear resampling is near-lossless


def test_parallax_pair_respects_depth():
    """Near pixels ( depth=1 ) must displace more than far ones ( depth=0 )."""
    h, w = 40, 64
    image = np.zeros((h, w, 3), np.uint8)
    image[:, : w // 3] = (255, 0, 0)    # left third red ( far )
    image[:, w // 3:] = (0, 0, 255)     # right two-thirds blue ( near )
    depth = np.zeros((h, w), np.float32)
    depth[:, w // 3:] = 1.0

    frame2 = synthesize_parallax_pair(image, depth, (10.0, 0.0))
    # near region shifts by +10px: red leaks into the first ~10 near columns
    y = h // 2
    leaked = frame2[y, w // 3 + 2: w // 3 + 12].astype(float)
    original = image[y, w // 3 + 2: w // 3 + 12].astype(float)
    # original: blue-dominant; warped: red-dominant
    assert leaked[:, 0].mean() > leaked[:, 2].mean()
    assert original[:, 2].mean() > original[:, 0].mean()


def test_raft_flow_rescales_to_native_resolution(painting, monkeypatch):
    """Mock torchvision RAFT returning a fixed (2, 520, 960) flow and check
    the wrapper resizes + rescales vectors back to painting pixels."""
    import torch

    class _T:
        def __call__(self, a, b):
            return a, b

    def fake_model(x1, x2):
        # constant flow of ( 4, 2 ) in the 520x960 canvas
        flow = torch.zeros(1, 2, 520, 960)
        flow[:, 0] = 4.0
        flow[:, 1] = 2.0
        return [flow]

    h, w = painting.shape[:2]
    flow = _raft_flow(fake_model, _T(), painting, painting, "cpu")
    assert flow.shape == (h, w, 2)
    sx, sy = w / 960.0, h / 520.0
    assert np.allclose(flow[..., 0], 4.0 * sx, atol=0.35)
    assert np.allclose(flow[..., 1], 2.0 * sy, atol=0.35)


def test_run_raft_analytic_fallback(painting, flat_depth, tmp_path, monkeypatch):
    """When RAFT weights are unavailable the analytic fallback still writes
    a full flow bundle, with magnitude == camera shift * depth."""
    monkeypatch.setattr(optical_flow, "load_raft", lambda prefer, device: (None, None, None))
    manifest = run_raft(painting, tmp_path, device="cpu", prefer="large", force=True)
    assert manifest["engine"] == "analytic_parallax"

    flow = np.load(tmp_path / "flow" / "flow.npy")
    assert flow.shape == painting.shape[:2] + (2,)
    shift = np.hypot(0.04 * painting.shape[1], 0.015 * painting.shape[0])
    expected_mag = shift * flat_depth  # |camera shift| * depth
    mag = np.linalg.norm(flow, axis=2)
    np.testing.assert_allclose(mag, expected_mag, atol=1e-3)
    assert (tmp_path / "flow" / "flow.png").exists()
    assert (tmp_path / "flow" / "flow.flo").exists()
    assert (tmp_path / "flow" / "flow_viz.png").exists()
    assert manifest["scale"] > 0
