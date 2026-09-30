"""Depth stage: prediction normalisation, smoothing, layer stats - with the
heavy models monkeypatched so nothing is downloaded."""

import numpy as np
import pytest
from PIL import Image

from inference import depth_estimate
from inference.depth_estimate import _predict, run_depth, smooth_depth


def test_smooth_depth_removes_speckle(flat_depth):
    noisy = flat_depth + np.random.default_rng(1).normal(0, 0.15, flat_depth.shape)
    noisy = np.clip(noisy, 0, 1).astype(np.float32)
    smoothed = smooth_depth(noisy)
    assert smoothed.shape == noisy.shape
    assert np.std(smoothed) < np.std(noisy)
    assert 0.0 <= smoothed.min() and smoothed.max() <= 1.0


def test_predict_midas_normalisation(painting, monkeypatch):
    """Monkeypatch torch + a fake MiDaS model returning a relative map."""
    import torch

    def transform(arr):
        import torch

        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float()

    class FakeMidas(torch.nn.Module):
        def forward(self, x):
            # "closer at the bottom": a vertical ramp, larger = nearer
            h = x.shape[-2]
            ramp = torch.linspace(0.2, 0.9, h).view(1, h, 1).expand(1, h, x.shape[-1])
            return ramp

    near = _predict((FakeMidas().eval(), transform), "midas_small", painting, "cpu")
    assert near.shape == painting.shape[:2]
    assert near.dtype == np.float32
    # near = 1 convention: bottom ( "closer" ) must beat top
    assert near[-1, 0] > near[0, 0]
    assert 0.0 <= near.min() and near.max() <= 1.0 + 1e-6


def test_run_depth_persists_manifest(painting, tmp_path, monkeypatch):
    """run_depth with a stubbed loader writes the full asset set + stats."""
    import torch

    class FakeMidas(torch.nn.Module):
        def forward(self, x):
            h = x.shape[-2]
            ramp = torch.linspace(0.1, 0.8, h).view(1, h, 1).expand(1, h, x.shape[-1])
            return ramp

    bundle = (FakeMidas().eval(), lambda arr: arr)
    monkeypatch.setattr(depth_estimate, "load_depth_model", lambda p, d: (bundle, "midas_small"))
    monkeypatch.setattr(
        depth_estimate, "_predict",
        lambda model, engine, img, dev: np.linspace(0, 1, img.shape[0])[:, None].repeat(img.shape[1], 1).astype(np.float32),
    )

    manifest = run_depth(painting, tmp_path, profile="midas", device="cpu", force=True)
    assert manifest["engine"] == "midas_small"
    assert manifest["nearIsOne"] is True
    assert (tmp_path / "depth" / "depth.png").exists()
    assert (tmp_path / "depth" / "depth.npy").exists()
    assert (tmp_path / "depth" / "depth_viz.png").exists()
    assert manifest["stats"]["min"] == pytest.approx(0.0, abs=0.01)

    # rerun without force: cache hit, same manifest
    manifest2 = run_depth(painting, tmp_path, profile="midas", device="cpu")
    assert manifest2 == manifest


def test_layer_depth_medians(painting, tmp_path, monkeypatch):
    """With layers.json present, per-layer medians land in the manifest."""
    import json

    layers_dir = tmp_path / "layers"
    layers_dir.mkdir(parents=True)
    mask = np.zeros(painting.shape[:2], np.uint8)
    mask[60:] = 255
    Image.fromarray(np.dstack([painting, mask])).save(layers_dir / "village.png")
    with open(tmp_path / "layers.json", "w") as fh:
        json.dump([{"name": "village", "file": "layers/village.png"}], fh)

    monkeypatch.setattr(depth_estimate, "load_depth_model", lambda p, d: (object(), "midas_small"))
    monkeypatch.setattr(
        depth_estimate, "_predict",
        lambda model, engine, img, dev: np.linspace(0, 1, img.shape[0])[:, None].repeat(img.shape[1], 1).astype(np.float32),
    )
    manifest = run_depth(painting, tmp_path, profile="midas", device="cpu", force=True)
    assert "village" in manifest["layerDepth"]
    assert manifest["layerDepth"]["village"] > 0.5  # bottom half is nearer
