"""Flow codec: PNG roundtrip, Middlebury .flo, colour wheel."""

import numpy as np
import pytest

from inference.flow_codec import (
    decode_flow_png,
    encode_flow_png,
    flow_to_color,
    read_flo,
    write_flo,
)


def _random_flow(h=24, w=32, scale=12.0, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(h, w, 2)).astype(np.float32)


def test_png_roundtrip_error_bounded():
    flow = _random_flow()
    rgb, scale = encode_flow_png(flow)
    recovered = decode_flow_png(rgb, scale)
    err = np.abs(recovered - flow)
    # 8-bit quantisation over a +-scale range: <= scale/255 per axis
    assert err.max() <= scale / 255.0 + 1e-4
    assert err.max() < 0.2


def test_png_scale_is_robust_to_outliers():
    flow = _random_flow(scale=5.0)
    flow[0, 0] = (1000.0, -1000.0)  # extreme outlier must not blow the scale
    _, scale = encode_flow_png(flow)
    assert scale < 50.0


def test_flo_roundtrip(tmp_path):
    flow = _random_flow()
    path = write_flo(tmp_path / "f.flo", flow)
    back = read_flo(path)
    np.testing.assert_array_almost_equal(back, flow)


def test_flo_header_layout(tmp_path):
    flow = np.zeros((10, 12, 2), np.float32)
    path = tmp_path / "f.flo"
    write_flo(path, flow)
    raw = np.fromfile(path, dtype=np.float32)
    assert raw[0] == pytest.approx(202021.253)  # Middlebury magic
    assert raw[1] == 12 and raw[2] == 10


def test_color_wheel_output_shape_and_range():
    flow = _random_flow()
    viz = flow_to_color(flow)
    assert viz.shape == flow.shape[:2] + (3,)
    assert viz.dtype == np.uint8
    assert viz.min() >= 0 and viz.max() <= 255
