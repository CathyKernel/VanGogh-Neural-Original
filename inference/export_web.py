"""Stage 5 - WebGL asset bundle export.

Copies everything the live renderer needs into ``renderer/web/assets/<scene>``:

    original.png     the untouched painting ( colour-preserving base canvas )
    layers/*.png     RGBA semantic layers in paint order
    depth.png        8-bit relative depth ( near = 1 = white )
    flow.png         invertible 8-bit RGB flow codec
    flow_viz.png     Middlebury colour wheel preview
    strokes.bin/.json Hertzmann brush stroke segments for instanced rendering
    manifest.json    machine-readable scene description

Also maintains ``assets/scenes.json`` ( the scene registry the renderer's
scene switcher reads ) and syncs the GLSL sources from ``renderer/shaders``
into ``renderer/web/shaders`` so the Netlify publish directory stays
self-contained.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from . import utils
from .utils import ensure_dir, get_logger, atomic_write_json, read_json

LOGGER = get_logger()


def _copy_if_present(src: Path, dst: Path) -> bool:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False


def _dominant_palette(rgb: np.ndarray, k: int = 6) -> List[List[int]]:
    """k-means dominant colours ( used by the renderer UI for theming )."""
    import cv2

    small = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 16, 0.8)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(-counts)
    return [[int(round(c)) for c in centers[i]] for i in order]


def sync_shaders(repo_root_dir: Path) -> int:
    """Copy GLSL sources into ``renderer/web/shaders`` ( published dir )."""
    src_dir = repo_root_dir / "renderer" / "shaders"
    dst_dir = repo_root_dir / "renderer" / "web" / "shaders"
    if not src_dir.exists():
        return 0
    ensure_dir(dst_dir)
    count = 0
    for shader in sorted(src_dir.glob("*")):
        if shader.suffix in (".glsl", ".vert", ".frag"):
            shutil.copy2(shader, dst_dir / shader.name)
            count += 1
    if count:
        LOGGER.info("Synced %d shader files into renderer/web/shaders", count)
    return count


def export_assets(scene_dir: Path | str, web_assets_dir: Path | str,
                  scene_name: str, image, models_used: Optional[Dict[str, str]] = None,
                  force: bool = False) -> dict:
    """Export one scene bundle.  Returns the scene manifest."""
    scene_dir = Path(scene_dir)
    out_dir = ensure_dir(Path(web_assets_dir) / scene_name)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        LOGGER.info("manifest.json exists for scene '%s' - skipping export", scene_name)
        return read_json(manifest_path)

    if isinstance(image, (str, Path)):
        image_rgb = utils.load_image_rgb(image)
    else:
        image_rgb = np.asarray(image)
    h, w = image_rgb.shape[:2]

    # ---- base image ( never recoloured - the whole point of the demo ) ----
    Image.fromarray(np.ascontiguousarray(image_rgb)).save(out_dir / "original.png")

    # ---- layers ----
    layers_manifest: List[dict] = read_json(scene_dir / "layers.json")
    for entry in layers_manifest:
        _copy_if_present(scene_dir / entry["file"], out_dir / f"layers/{Path(entry['file']).name}")

    # ---- depth ----
    depth_fragment = read_json(scene_dir / "depth" / "depth.json")
    _copy_if_present(scene_dir / depth_fragment["file"], out_dir / "depth.png")
    _copy_if_present(scene_dir / depth_fragment.get("vizFile", ""), out_dir / "depth_viz.png")

    # ---- flow ----
    flow_fragment = read_json(scene_dir / "flow" / "flow.json")
    _copy_if_present(scene_dir / flow_fragment["file"], out_dir / "flow.png")
    _copy_if_present(scene_dir / flow_fragment.get("vizFile", ""), out_dir / "flow_viz.png")

    # ---- strokes ----
    strokes_fragment = read_json(scene_dir / "strokes" / "strokes.json")
    _copy_if_present(scene_dir / strokes_fragment["bin"], out_dir / "strokes.bin")

    # ---- manifest ----
    layer_depth: Dict[str, float] = depth_fragment.get("layerDepth", {})
    manifest = {
        "scene": scene_name,
        "title": scene_name.replace("_", " ").title(),
        "width": w,
        "height": h,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models_used or {},
        "original": "original.png",
        "layers": [
            {
                "name": entry["name"],
                "file": f"layers/{Path(entry['file']).name}",
                "paintOrder": idx,
                "areaFrac": entry.get("areaFrac", 0.0),
                "meanColor": entry.get("meanColor", [0, 0, 0]),
                "depth": layer_depth.get(entry["name"], 0.5),
            }
            for idx, entry in enumerate(layers_manifest)
        ],
        "depth": {
            "file": "depth.png",
            "vizFile": "depth_viz.png",
            "nearIsOne": True,
            "engine": depth_fragment.get("engine", "unknown"),
            "stats": depth_fragment.get("stats", {}),
        },
        "flow": {
            "file": "flow.png",
            "vizFile": "flow_viz.png",
            "scale": flow_fragment.get("scale", 1.0),
            "cameraShift": flow_fragment.get("cameraShift", [0, 0]),
            "engine": flow_fragment.get("engine", "unknown"),
            "stats": flow_fragment.get("stats", {}),
        },
        "strokes": {
            "bin": "strokes.bin",
            "count": strokes_fragment.get("count", 0),
            "strideFloats": strokes_fragment.get("strideFloats", 12),
            "radiusRange": strokes_fragment.get("radiusRange", [0, 0]),
            "lengthRange": strokes_fragment.get("lengthRange", [0, 0]),
            "layers": strokes_fragment.get("layers", []),
        },
        "palette": {"dominant": _dominant_palette(image_rgb)},
    }
    atomic_write_json(manifest_path, manifest)
    LOGGER.info("Exported scene '%s' -> %s ( %d layers, %d strokes )",
                scene_name, out_dir, len(manifest["layers"]), manifest["strokes"]["count"])
    return manifest


def update_scene_registry(web_assets_dir: Path | str) -> List[dict]:
    """Rebuild ``scenes.json`` from the manifests present in the assets dir."""
    assets_dir = Path(web_assets_dir)
    scenes: List[dict] = []
    for manifest_path in sorted(assets_dir.glob("*/manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:  # noqa: BLE001 - corrupt manifest tolerated
            LOGGER.warning("Skipping unreadable manifest %s ( %s )", manifest_path, exc)
            continue
        scenes.append({
            "scene": manifest["scene"],
            "title": manifest.get("title", manifest["scene"]),
            "manifest": f"{manifest_path.parent.name}/manifest.json",
            "thumbnail": f"{manifest_path.parent.name}/original.png",
            "strokes": manifest.get("strokes", {}).get("count", 0),
            "layers": len(manifest.get("layers", [])),
        })
    if scenes:
        atomic_write_json(assets_dir / "scenes.json", scenes)
        LOGGER.info("Scene registry: %d scene(s)", len(scenes))
    return scenes


# ---------------------------------------------------------------------------
# Backwards-compatible entry point ( original skeleton called export_assets() )
# ---------------------------------------------------------------------------

def export_web(scene_dir: Path | str, web_assets_dir: Path | str, scene_name: str,
               image, models_used: Optional[Dict[str, str]] = None,
               force: bool = False) -> dict:
    """Full export pass: assets + registry + shader sync."""
    manifest = export_assets(scene_dir, web_assets_dir, scene_name, image,
                             models_used=models_used, force=force)
    update_scene_registry(web_assets_dir)
    sync_shaders(utils.repo_root())
    return manifest
