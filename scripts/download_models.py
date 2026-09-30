#!/usr/bin/env python3
"""Download model checkpoints for the Van Gogh pipeline.

SAM checkpoints are fetched from Meta's CDN with resume support.
ZoeDepth / MiDaS and RAFT weights are pulled lazily by torch.hub /
torchvision on first inference - pass --warmup to pre-download them now.

Examples
--------
    python scripts/download_models.py --profile cpu
    python scripts/download_models.py --profile paper --warmup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import model_zoo, utils  # noqa: E402


def warmup_depth(profile: str, device: str) -> str:
    from inference import depth_estimate

    _model, engine = depth_estimate.load_depth_model(profile, device)
    depth_estimate.release_models()
    return engine


def warmup_raft(prefer: str, device: str) -> str:
    from inference import optical_flow

    model, _t, engine = optical_flow.load_raft(prefer, device)
    if model is None:
        return "unavailable"
    optical_flow.release_models()
    return engine or "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--profile", choices=["paper", "cpu", "auto"], default="auto")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", action="store_true",
                        help="also pre-download torch.hub depth + torchvision RAFT weights")
    args = parser.parse_args()

    utils.setup_logging()
    engines = model_zoo.ensure_profile(args.profile, ROOT / args.ckpt_dir, args.device,
                                       allow_download=True)
    print(f"\nSAM checkpoint : {engines['sam']}")
    print(f"Depth engine   : {engines['depth']} ( torch.hub, lazy )")
    print(f"Flow engine    : {engines['raft']} ( torchvision, lazy )")

    if args.warmup:
        print("\nWarming up depth model ...")
        print(f"  -> {warmup_depth(engines['depth'], args.device)}")
        print("Warming up RAFT ...")
        print(f"  -> {warmup_raft(engines['raft'], args.device)}")

    print("\nManual URLs ( if you prefer ):")
    for variant, info in model_zoo.SAM_CHECKPOINTS.items():
        print(f"  SAM {variant}: {info['url']}")
    print("  ZoeDepth : https://github.com/isl-org/ZoeDepth ( torch.hub )")
    print("  MiDaS    : https://github.com/isl-org/MiDaS ( torch.hub )")
    print("  RAFT     : torchvision.models.optical_flow ( weights via download.pytorch.org )")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
