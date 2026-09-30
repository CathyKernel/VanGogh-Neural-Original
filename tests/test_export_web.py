"""Web export: manifest structure, scene registry, shader sync."""

import json

import numpy as np
import pytest
from PIL import Image

from inference import utils
from inference.export_web import export_assets, export_web, sync_shaders, update_scene_registry


def _build_fake_scene(root: "Path", painting: np.ndarray) -> "Path":
    scene = root / "scene"
    (scene / "layers").mkdir(parents=True)
    (scene / "depth").mkdir(parents=True)
    (scene / "flow").mkdir(parents=True)
    (scene / "strokes").mkdir(parents=True)

    alpha = np.full(painting.shape[:2], 255, np.uint8)
    Image.fromarray(np.dstack([painting, alpha])).save(scene / "layers" / "sky.png")
    with open(scene / "layers.json", "w") as fh:
        json.dump([{"name": "sky", "paintOrder": 0, "file": "layers/sky.png",
                    "areaFrac": 1.0, "bbox": [0, 0, 1, 1], "meanColor": [30, 60, 120]}], fh)

    gray = (np.zeros(painting.shape[:2], np.uint8))
    Image.fromarray(gray).save(scene / "depth" / "depth.png")
    Image.fromarray(painting).save(scene / "depth" / "depth_viz.png")
    with open(scene / "depth" / "depth.json", "w") as fh:
        json.dump({"engine": "midas_small", "file": "depth/depth.png",
                   "vizFile": "depth/depth_viz.png", "bitDepth": 8, "nearIsOne": True,
                   "layerDepth": {"sky": 0.5}, "stats": {}}, fh)

    Image.fromarray(painting).save(scene / "flow" / "flow.png")
    Image.fromarray(painting).save(scene / "flow" / "flow_viz.png")
    with open(scene / "flow" / "flow.json", "w") as fh:
        json.dump({"engine": "raft_large", "file": "flow/flow.png",
                   "vizFile": "flow/flow_viz.png", "scale": 12.5,
                   "cameraShift": [30.7, 7.2], "stats": {}}, fh)

    (np.zeros((4, 12), np.float32)).tofile(scene / "strokes" / "strokes.bin")
    with open(scene / "strokes" / "strokes.json", "w") as fh:
        json.dump({"count": 4, "strideFloats": 12, "bin": "strokes/strokes.bin",
                   "layers": ["sky"], "radiusRange": [3.0, 9.0], "lengthRange": [2.0, 12.0]}, fh)
    return scene


def test_export_assets_manifest(tmp_path, painting):
    scene = _build_fake_scene(tmp_path, painting)
    web = tmp_path / "web" / "assets"

    manifest = export_assets(scene, web, "fake", painting,
                             models_used={"sam": "sam_vit_b", "depth": "midas_small",
                                          "flow": "raft_large"})
    out = web / "fake"
    assert (out / "manifest.json").exists()
    assert (out / "original.png").exists()
    assert (out / "layers" / "sky.png").exists()
    assert (out / "depth.png").exists()
    assert (out / "flow.png").exists()
    assert (out / "strokes.bin").exists()

    assert manifest["scene"] == "fake"
    assert manifest["width"] == painting.shape[1]
    assert manifest["height"] == painting.shape[0]
    assert manifest["models"]["sam"] == "sam_vit_b"
    assert manifest["flow"]["scale"] == 12.5
    assert manifest["strokes"]["count"] == 4
    assert manifest["layers"][0]["depth"] == 0.5
    assert len(manifest["palette"]["dominant"]) == 6
    for color in manifest["palette"]["dominant"]:
        assert len(color) == 3 and 0 <= min(color) and max(color) <= 255

    # idempotent: rerun without force hits the cache
    manifest2 = export_assets(scene, web, "fake", painting)
    assert manifest2 == manifest


def test_scene_registry(tmp_path, painting):
    scene = _build_fake_scene(tmp_path, painting)
    web = tmp_path / "web" / "assets"
    export_assets(scene, web, "fake", painting)

    scenes = update_scene_registry(web)
    assert len(scenes) == 1
    assert scenes[0]["scene"] == "fake"
    assert scenes[0]["manifest"] == "fake/manifest.json"
    assert scenes[0]["strokes"] == 4
    registry = json.loads((web / "scenes.json").read_text())
    assert registry[0]["title"] == "Fake"


def test_sync_shaders_copies_into_web(tmp_path, monkeypatch):
    root = tmp_path
    src = root / "renderer" / "shaders"
    src.mkdir(parents=True)
    (src / "common.glsl").write_text("// test")
    (src / "depth_parallax.frag").write_text("// test")
    (src / "notes.txt").write_text("ignored")

    monkeypatch.setattr(utils, "repo_root", lambda: root)
    count = sync_shaders(root)
    assert count == 2
    assert (root / "renderer" / "web" / "shaders" / "common.glsl").exists()
    assert not (root / "renderer" / "web" / "shaders" / "notes.txt").exists()


def test_export_web_full_pass(tmp_path, painting, monkeypatch):
    scene = _build_fake_scene(tmp_path, painting)
    root = tmp_path
    (root / "renderer" / "shaders").mkdir(parents=True)
    (root / "renderer" / "shaders" / "brush.frag").write_text("// brush")
    monkeypatch.setattr(utils, "repo_root", lambda: root)

    web = root / "renderer" / "web" / "assets"
    manifest = export_web(scene, web, "fake", painting)
    assert (web / "fake" / "manifest.json").exists()
    assert (web / "scenes.json").exists()
    assert (root / "renderer" / "web" / "shaders" / "brush.frag").exists()
    assert manifest["scene"] == "fake"
