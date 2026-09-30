"""Model checkpoint discovery / download for the SAM stage.

ZoeDepth, MiDaS and RAFT weights are fetched lazily by ``torch.hub`` /
``torchvision`` on first use, so only SAM checkpoints need explicit
management here.  Downloads are resumable ( HTTP Range ) and size-checked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from .utils import bytes_human, get_logger

LOGGER = get_logger()

# name -> ( url, filename, expected_bytes or None )
SAM_CHECKPOINTS: Dict[str, Dict[str, object]] = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "file": "sam_vit_h_4b8939.pth",
        "bytes": 2564550879,
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "file": "sam_vit_l_0b3195.pth",
        "bytes": 1246251137,
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "file": "sam_vit_b_01ec64.pth",
        "bytes": 375042383,
    },
}


def checkpoint_path(model_type: str, ckpt_dir: Path | str) -> Path:
    return Path(ckpt_dir) / SAM_CHECKPOINTS[model_type]["file"]


def have_checkpoint(model_type: str, ckpt_dir: Path | str) -> bool:
    path = checkpoint_path(model_type, ckpt_dir)
    expected = SAM_CHECKPOINTS[model_type]["bytes"]
    return path.exists() and (expected is None or path.stat().st_size == expected)


def download_sam(model_type: str, ckpt_dir: Path | str, timeout: int = 120) -> Optional[Path]:
    """Download a SAM checkpoint with resume support.  Returns the path."""
    import requests

    info = SAM_CHECKPOINTS[model_type]
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / info["file"]
    url = info["url"]
    expected = info["bytes"]

    if have_checkpoint(model_type, ckpt_dir):
        LOGGER.info("SAM %s checkpoint already present (%s)", model_type, target.name)
        return target

    part = target.with_suffix(target.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    LOGGER.info("Downloading SAM %s (%s) from %s", model_type, bytes_human(expected or 0), url)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            if offset and resp.status_code != 206:
                LOGGER.warning("Server ignored resume request - restarting download")
                offset = 0
                part.unlink(missing_ok=True)
            resp.raise_for_status()
            mode = "ab" if offset else "wb"
            done = offset
            last_log = -10.0
            import time

            t0 = time.time()
            with open(part, mode) as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if expected and time.time() - last_log > 10.0:
                        pct = done / expected * 100.0
                        rate = (done - offset) / max(time.time() - t0, 1e-3) / (1 << 20)
                        LOGGER.info("  ... %5.1f%%  (%.1f MB/s)", pct, rate)
                        last_log = time.time()
        if expected is not None and done != expected:
            LOGGER.error("Size mismatch after download: got %d, expected %d", done, expected)
            return None
        os.replace(part, target)
        LOGGER.info("SAM %s checkpoint stored at %s", model_type, target)
        return target
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.error("Download failed for SAM %s: %s", model_type, exc)
        return None


def ensure_profile(profile: str, ckpt_dir: Path | str, device: str = "auto",
                   allow_download: bool = True) -> Dict[str, str]:
    """Make sure the SAM checkpoint required by *profile* is available.

    Returns a mapping describing what was resolved, e.g.::

        {"sam": "vit_b", "depth": "midas", "raft": "large", "downloaded": "vit_b"}

    Depth / flow engines load lazily through torch.hub / torchvision, so only
    their *names* are resolved here.
    """
    from . import sam_segment

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else (Path(__file__).resolve().parents[1] / "checkpoints")
    want_gpu = device == "cuda" or (device == "auto" and _cuda_available())
    if profile == "paper" or (profile == "auto" and want_gpu):
        sam_type, depth, raft = "vit_h", "zoedepth", "large"
    else:
        sam_type, depth, raft = "vit_b", "midas", "large"

    resolved: Dict[str, str] = {"sam": "auto", "depth": depth, "raft": raft}

    # degrade the SAM tier if the paper-level file cannot be obtained
    for candidate in (sam_type, "vit_l", "vit_b"):
        if have_checkpoint(candidate, ckpt_dir):
            resolved["sam"] = candidate
            break
        if allow_download and download_sam(candidate, ckpt_dir):
            resolved["sam"] = candidate
            resolved["downloaded"] = candidate
            break
    else:
        resolved["sam"] = "classical"
    if resolved["sam"] == "classical":
        LOGGER.warning("Proceeding without SAM checkpoints - classical k-means fallback")
    return resolved


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False
