"""Painterly stroke generation: record layout, layer budgets, colours."""

import numpy as np

from inference.painterly import (
    SEGMENT_STRIDE,
    generate_stroke_segments,
    run_strokes,
)

RECORD = ["x", "y", "dirx", "diry", "len", "radius", "opacity", "layer", "r", "g", "b", "phase"]


def test_record_layout_matches_shader_attributes(painting):
    """The 12 floats must group exactly as brush.vert declares its vec4s:
    aSegA = x,y,dirx,diry | aSegB = len,radius,opacity,layer | aSegC = r,g,b,phase
    """
    h, w = painting.shape[:2]
    masks = {"sky": np.ones((h, w), bool)}
    segments, meta = generate_stroke_segments(painting, masks, ["sky"], rng_seed=5)
    assert segments.ndim == 2 and segments.shape[1] == SEGMENT_STRIDE == 12

    x, y, dx, dy, ln, radius, opacity, layer, r, g, b, phase = segments.T

    # geometry sanity
    assert x.min() >= 0 and x.max() <= w
    assert y.min() >= 0 and y.max() <= h
    assert radius.min() > 0
    assert ln.min() > 0
    # directions are unit vectors
    norms = np.hypot(dx, dy)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    # paint fields in range
    assert 0 <= opacity.min() and opacity.max() <= 1
    assert 0 <= r.min() and r.max() <= 1
    assert 0 <= phase.min() and phase.max() <= 2 * np.pi
    # single layer -> every record paints layer 0
    assert np.all(layer == 0)


def test_per_layer_budgets_share_the_cap(painting):
    """The segment budget must be split across layers, not eaten by layer 0."""
    h, w = painting.shape[:2]
    top = np.zeros((h, w), bool)
    top[: h // 2] = True
    bottom = ~top
    masks = {"sky": top, "village": bottom}
    segments, meta = generate_stroke_segments(painting, masks, ["sky", "village"],
                                              max_segments=4000, rng_seed=5)
    layers_used = np.unique(segments[:, 7])
    assert set(layers_used.tolist()) == {0.0, 1.0}, "both layers must receive strokes"
    assert meta["count"] <= 4000 + 2 * 1500  # cap + small min-budget allowance
    per_layer = meta["segmentsPerLayer"]
    assert per_layer["sky"] > 0 and per_layer["village"] > 0


def test_stroke_colors_track_source_image(painting):
    """Stroke colours must be sampled from the painting ( preservation )."""
    h, w = painting.shape[:2]
    masks = {"sky": np.ones((h, w), bool)}
    segments, _ = generate_stroke_segments(painting, masks, ["sky"], rng_seed=5)
    mean_stroke = segments[:, 8:11].mean(axis=0)
    mean_image = (painting.reshape(-1, 3).mean(axis=0) / 255.0)
    np.testing.assert_allclose(mean_stroke, mean_image, atol=0.06)


def test_run_strokes_end_to_end(painting, tmp_path):
    """Full stage: writes strokes.bin ( stride 12 ) + strokes.json metadata."""
    from PIL import Image
    import json

    layers_dir = tmp_path / "layers"
    layers_dir.mkdir(parents=True)
    alpha = np.full(painting.shape[:2], 255, np.uint8)
    Image.fromarray(np.dstack([painting, alpha])).save(layers_dir / "sky.png")
    with open(tmp_path / "layers.json", "w") as fh:
        json.dump([{"name": "sky", "file": "layers/sky.png"}], fh)

    meta = run_strokes(painting, tmp_path, max_segments=3000, force=True)
    assert meta["strideFloats"] == 12
    assert meta["count"] > 0

    raw = np.fromfile(tmp_path / "strokes" / "strokes.bin", dtype=np.float32)
    assert raw.size == meta["count"] * 12
    assert raw.size % 12 == 0

    # cached rerun: same manifest, no regeneration
    meta2 = run_strokes(painting, tmp_path, max_segments=3000)
    assert meta2 == meta
