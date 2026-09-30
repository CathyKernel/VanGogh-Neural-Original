"""Semantic labelling heuristics and layer-stack assembly ( no SAM needed )."""

import numpy as np
from PIL import Image

from inference.sam_segment import (
    LAYER_PAINT_ORDER,
    blob_assist,
    build_layer_stack,
    classify_mask,
    label_sam_masks,
    layer_specs_to_json,
    _mask_stats,
)


def _mask_from_region(shape, x0, y0, x1, y1):
    m = np.zeros(shape[:2], bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_stats_geometry(painting):
    m = _mask_from_region(painting.shape, 0, 0, 64, 48)
    s = _mask_stats(painting, m)
    assert 0 < s["area_frac"] < 1
    assert s["cx"] < 0.55 and s["cy"] < 0.55
    assert s["bw"] == 0.5 and s["bh"] == 0.5
    assert s["fill"] == 1.0


def test_moon_classification(painting):
    # the synthetic moon lives at the upper-right ( seed fixture )
    m = _mask_from_region(painting.shape, 94, 12, 106, 26)
    assert classify_mask(_mask_stats(painting, m)) == "moon"


def test_village_classification(painting):
    m = _mask_from_region(painting.shape, 16, 80, 100, 94)
    assert classify_mask(_mask_stats(painting, m)) == "village"


def test_tree_classification(painting):
    m = _mask_from_region(painting.shape, 0, 24, 10, 90)
    assert classify_mask(_mask_stats(painting, m)) == "tree"


def test_label_sam_masks_covers_everything(painting):
    masks = [
        {"segmentation": _mask_from_region(painting.shape, 94, 12, 106, 26), "area": 100},
        {"segmentation": _mask_from_region(painting.shape, 16, 80, 100, 94), "area": 900},
        {"segmentation": _mask_from_region(painting.shape, 0, 24, 10, 90), "area": 500},
    ]
    layers = label_sam_masks(painting, masks, labels_mode="auto")
    covered = np.zeros(painting.shape[:2], bool)
    for mask in layers.values():
        covered |= mask
    assert covered.all(), "every pixel must belong to some layer ( sky backfills )"
    assert "sky" in layers
    assert "moon" in layers
    # paint order sanity: sky behind moon
    assert LAYER_PAINT_ORDER["sky"] < LAYER_PAINT_ORDER["moon"]


def test_blob_assist_recovers_stars(painting):
    layers = {"sky": np.ones(painting.shape[:2], bool)}
    blob_assist(painting, layers)
    assert "stars" in layers and layers["stars"].sum() > 0
    assert "moon" in layers and layers["moon"].sum() > 0
    # stars never overlap the moon halo
    assert not (layers["stars"] & layers["moon"]).any()


def test_build_layer_stack_order_and_json(painting, tmp_path):
    h, w = painting.shape[:2]
    layers = {
        "sky": np.ones((h, w), bool),
        "village": _mask_from_region(painting.shape, 0, 72, 128, 96),
        "moon": _mask_from_region(painting.shape, 92, 8, 108, 28),
    }
    specs = build_layer_stack(painting, layers)
    assert [s.name for s in specs] == ["sky", "village", "moon"]
    assert [s.paint_order for s in specs] == [0, 1, 2]

    manifest = layer_specs_to_json(specs, painting.shape)
    assert manifest[0]["name"] == "sky"
    assert manifest[0]["file"] == "layers/sky.png"
    assert manifest[2]["meanColor"] == list(specs[2].mean_color)

    # paint order monotonic in the manifest
    orders = [entry["paintOrder"] for entry in manifest]
    assert orders == sorted(orders)
