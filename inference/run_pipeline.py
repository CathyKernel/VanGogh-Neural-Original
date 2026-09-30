"""Pipeline orchestrator and command-line interface.

Runs the four CV stages plus the web export for one or more input images::

    SAM semantic layers -> ZoeDepth/MiDaS depth -> RAFT optical flow
    -> Hertzmann brush strokes -> WebGL asset export

Usage
-----
Run the default demo ( every image under ``data/input`` )::

    python -m inference.run_pipeline

Run a specific painting with the paper-level models on CUDA::

    python -m inference.run_pipeline --input data/input/starry_night.png \
        --profile paper --device cuda

Each stage is cached: reruns skip stages whose JSON manifest already exists
unless ``--force`` is given.  Stage selection via ``--stages sam,flow``.

The script also works when executed directly from inside ``inference/``::
    python inference/run_pipeline.py            ( flat-module compatibility )
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Import bootstrap: support both ``python -m inference.run_pipeline`` and
# ``python inference/run_pipeline.py`` ( the original skeleton's style ).
# ---------------------------------------------------------------------------
try:
    from . import utils
    from . import sam_segment, depth_estimate, optical_flow, painterly, export_web, model_zoo
except ImportError:  # pragma: no cover - direct execution fallback
    _PKG_PARENT = str(Path(__file__).resolve().parents[1])
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    from inference import utils  # type: ignore
    from inference import (  # type: ignore
        sam_segment, depth_estimate, optical_flow, painterly, export_web, model_zoo,
    )

from inference.utils import get_logger  # type: ignore

LOGGER = get_logger()

STAGES = ("sam", "depth", "flow", "strokes", "export")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vangogh",
        description="Van Gogh paper-level neural-preserving pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", nargs="+", default=None, metavar="PATH[:SCENE]",
        help="input image(s); each may carry ':scene_name'. "
             "Default: every image in data/input/")
    parser.add_argument("--profile", choices=["paper", "cpu", "auto"], default="auto",
                        help="model profile: paper = SAM ViT-H + ZoeDepth-NK + RAFT-large, "
                             "cpu = SAM ViT-B + MiDaS-small + RAFT-large")
    parser.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    parser.add_argument("--max-side", type=int, default=768,
                        help="resize longest side to this many pixels")
    parser.add_argument("--points-per-side", type=int, default=None,
                        help="SAM point grid density ( default 32 GPU / 16 CPU )")
    parser.add_argument("--labels", choices=["auto", "generic"], default="auto",
                        help="semantic labelling: starry-night priors or generic layers")
    parser.add_argument("--output-root", default="outputs", help="intermediate output root")
    parser.add_argument("--web-root", default="renderer/web/assets",
                        help="renderer asset directory ( published on Netlify )")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="checkpoint directory")
    parser.add_argument("--stages", default="all",
                        help=f"comma list of stages to run: {','.join(STAGES)} or 'all'")
    parser.add_argument("--max-segments", type=int, default=48000,
                        help="cap on brush stroke segments per scene")
    parser.add_argument("--force", action="store_true", help="rerun stages ignoring caches")
    parser.add_argument("--no-download", action="store_true",
                        help="never download checkpoints automatically")
    parser.add_argument("--quiet", action="store_true", help="warning-only logging")
    return parser


def resolve_inputs(args: argparse.Namespace, root: Path) -> List[Dict[str, str]]:
    """Normalise --input entries ( with optional :scene suffix )."""
    items: List[Dict[str, str]] = []
    if args.input:
        for entry in args.input:
            if ":" in entry and not entry.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                path, scene = entry.rsplit(":", 1)
            elif ":" in entry:
                drive, rest = entry.split(":", 1) if sys.platform == "win32" else (None, entry)
                path, scene = (rest, Path(rest).stem) if drive is None else (entry, Path(entry).stem)
            else:
                path, scene = entry, Path(entry).stem
            path = Path(path)
            if not path.is_absolute():
                path = (root / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"input image not found: {path}")
            items.append({"path": str(path), "scene": scene or Path(path).stem})
        return items

    input_dir = root / "data" / "input"
    if not input_dir.exists():
        raise FileNotFoundError(
            f"no --input given and {input_dir} does not exist; "
            "run scripts/make_synthetic.py or fetch the Starry Night image first")
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            items.append({"path": str(path), "scene": path.stem})
    if not items:
        raise FileNotFoundError(f"no images found in {input_dir}")
    return items


# ---------------------------------------------------------------------------
# Per-scene execution
# ---------------------------------------------------------------------------

def _engine_from_json(fragment, engines: Dict[str, str], key: str, default: str) -> str:
    """Pull the engine name out of a stage manifest fragment, with fallbacks.

    Prefers the persisted stage manifest ( survives cached / partial reruns ),
    then the resolved engine table, then a human-readable default.
    """
    if isinstance(fragment, dict) and fragment.get("engine"):
        return str(fragment["engine"])
    if isinstance(fragment, list) and fragment and isinstance(fragment[0], dict) \
            and fragment[0].get("engine"):
        return str(fragment[0]["engine"])
    return str(engines.get(key, default))


def run_scene(path: str, scene: str, args: argparse.Namespace, root: Path,
              engines: Dict[str, str]) -> Dict:
    """Run all requested stages for one image.  Returns the scene summary."""
    timer = utils.StageTimer()
    scene_dir = utils.ensure_dir(root / args.output_root / scene)
    stages = set(STAGES) if args.stages == "all" else {s.strip() for s in args.stages.split(",")}
    unknown = stages - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    LOGGER.info("=" * 72)
    LOGGER.info("Scene '%s' <- %s", scene, path)

    # canonical resized input ( cached so reruns and export agree on size )
    cache_input = scene_dir / "input.png"
    if not cache_input.exists() or args.force:
        image_rgb = utils.load_image_rgb(path, max_side=args.max_side)
        utils.save_rgb_png(cache_input, image_rgb)
        LOGGER.info("Cached %dx%d working copy at %s", image_rgb.shape[1], image_rgb.shape[0],
                    cache_input)
    image_rgb = utils.load_image_rgb(cache_input)

    sam_variant = engines.get("sam", "auto")
    if sam_variant == "classical":
        sam_variant = "classical"

    layers_manifest: Optional[List[dict]] = None
    depth_manifest: Optional[dict] = None
    flow_manifest: Optional[dict] = None

    if "sam" in stages:
        with timer.stage("sam"):
            layers_manifest = sam_segment.run_sam(
                cache_input, scene_dir,
                model_type=sam_variant if sam_variant != "auto" else args.profile,
                ckpt_dir=root / args.ckpt_dir, device=args.device,
                points_per_side=args.points_per_side, labels_mode=args.labels,
                force=args.force)
    if "depth" in stages:
        with timer.stage("depth"):
            depth_manifest = depth_estimate.run_depth(
                cache_input, scene_dir, profile=engines.get("depth", "auto"),
                device=args.device, force=args.force)
    if "flow" in stages:
        with timer.stage("flow"):
            flow_manifest = optical_flow.run_raft(
                cache_input, scene_dir, device=args.device,
                prefer=engines.get("raft", "large"), force=args.force)
    if "strokes" in stages:
        with timer.stage("strokes"):
            painterly.run_strokes(cache_input, scene_dir,
                                  max_segments=args.max_segments, force=args.force)

    # engine names from the persisted stage manifests ( survives --force reruns
    # of individual stages and cached runs )
    if layers_manifest is None and (scene_dir / "layers.json").exists():
        layers_manifest = utils.read_json(scene_dir / "layers.json")
    if depth_manifest is None and (scene_dir / "depth" / "depth.json").exists():
        depth_manifest = utils.read_json(scene_dir / "depth" / "depth.json")
    if flow_manifest is None and (scene_dir / "flow" / "flow.json").exists():
        flow_manifest = utils.read_json(scene_dir / "flow" / "flow.json")
    models_used = {
        "sam": _engine_from_json(layers_manifest, engines, "sam", "vit_b"),
        "depth": _engine_from_json(depth_manifest, engines, "depth", "midas"),
        "flow": _engine_from_json(flow_manifest, engines, "raft", "raft_large"),
    }
    if "export" in stages:
        with timer.stage("export"):
            export_web.export_web(
                scene_dir, root / args.web_root, scene, cache_input,
                models_used=models_used, force=args.force)

    report = {
        "scene": scene,
        "input": path,
        "models": models_used,
        "stages_seconds": timer.as_dict(),
    }
    utils.atomic_write_json(scene_dir / "report.json", report)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        import logging

        utils.get_logger().setLevel(logging.WARNING)

    root = utils.repo_root()
    t0 = time.perf_counter()
    LOGGER.info("Van Gogh neural-preserving pipeline ( profile=%s, device=%s )",
                args.profile, args.device)

    engines = model_zoo.ensure_profile(args.profile, root / args.ckpt_dir, args.device,
                                       allow_download=not args.no_download)
    LOGGER.info("Resolved engines: sam=%s depth=%s raft=%s", engines["sam"],
                engines["depth"], engines["raft"])

    inputs = resolve_inputs(args, root)
    LOGGER.info("Processing %d scene(s): %s", len(inputs), ", ".join(i["scene"] for i in inputs))

    reports = []
    for item in inputs:
        reports.append(run_scene(item["path"], item["scene"], args, root, engines))

    # one GPU-RAM-friendly process per run: free anything left cached
    sam_segment.release_models()
    depth_estimate.release_models()
    optical_flow.release_models()

    LOGGER.info("=" * 72)
    LOGGER.info("Pipeline finished in %.1fs", time.perf_counter() - t0)
    for report in reports:
        models = report["models"]
        LOGGER.info("  %-22s sam=%-14s depth=%-12s flow=%-10s %s",
                    report["scene"], models.get("sam"), models.get("depth"),
                    models.get("flow"), report["stages_seconds"])
    LOGGER.info("Open renderer/web/index.html ( or serve it ) to view the live demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
