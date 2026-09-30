"""Stage 2 - Monocular depth estimation ( ZoeDepth with MiDaS fallback ).

Depth drives two things downstream:

* the **parallax pair synthesis** for RAFT ( stage 3 ) - a synthetic camera
  translation warps the painting by ``shift * depth`` so that the optical
  flow field encodes genuine 2.5D scene motion, and
* the **per-layer parallax amplitude** in the WebGL renderer ( sky drifts
  less than the cypress tree ), giving the "living painting" effect.

Model priority
--------------
1. ``ZoeD_NK`` (isl-org/ZoeDepth, torch.hub) - metric depth, paper-level.
2. ``MiDaS_small`` (intel-isl/MiDaS, torch.hub) - fast CPU fallback.

Both are normalised to a *relative* float32 map in [0, 1] where **1 = nearest**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from . import utils
from .utils import ensure_dir, get_logger, atomic_write_json

LOGGER = get_logger()

_DEPTH_MODEL_CACHE: Dict[str, object] = {}


def _ensure_hub_repo(owner_repo: str, branch: str = "master") -> None:
    """Pre-seed the torch.hub cache from a codeload tarball.

    ``git clone`` can be orders of magnitude slower than a plain tarball
    fetch in constrained networks; this makes hub loading deterministic.
    The cache layout mirrors what ``torch.hub`` itself uses, so a normal
    ``torch.hub.load(...)`` hits the cache afterwards.
    """
    import tarfile
    import tempfile
    import torch

    cache_dir = Path(torch.hub.get_dir()) / f"{owner_repo.replace('/', '_')}_{branch}"
    if (cache_dir / "hubconf.py").exists():
        return
    tar_url = f"https://codeload.github.com/{owner_repo}/tar.gz/refs/heads/{branch}"
    LOGGER.info("Fetching %s via %s", owner_repo, tar_url)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests

        with requests.get(tar_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    tmp.write(chunk)
                tmp_path = tmp.name
        with tarfile.open(tmp_path) as tar:
            members = [m for m in tar.getmembers() if "/" in m.name]
            root = members[0].name.split("/")[0]
            cache_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                member.name = member.name[len(root) + 1:]
                if member.name:
                    tar.extract(member, cache_dir)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Could not pre-seed %s ( %s ) - torch.hub will try git", owner_repo, exc)


def _hub_repo_dir(owner_repo: str, branch: str) -> Path:
    """Path of the pre-seeded hub cache for *owner_repo* ( seeds on demand )."""
    _ensure_hub_repo(owner_repo, branch)
    import torch

    return Path(torch.hub.get_dir()) / f"{owner_repo.replace('/', '_')}_{branch}"


class _offline_hub:
    """Context manager: route torch.hub.load calls to pre-seeded local copies.

    Some upstream code ( MiDaS's EfficientNet backbone builder ) issues its
    own ``torch.hub.load("rwightman/gen-efficientnet-pytorch", ...)`` - a
    remote git clone that can stall for a very long time on throttled
    networks.  When the repo is already seeded in the hub cache we load it
    from disk instead; unknown repos fall through to normal behaviour.
    """

    _EXTRA_REPOS = ("rwightman/gen-efficientnet-pytorch",)

    def __init__(self) -> None:
        self._orig = None

    def __enter__(self):
        import torch

        self._orig = torch.hub.load

        def patched(repo, model, *args, source="github", **kwargs):
            if source == "github" and isinstance(repo, str) and "/" in repo:
                for branch in ("master", "main"):
                    cache_dir = Path(torch.hub.get_dir()) / f"{repo.replace('/', '_')}_{branch}"
                    if (cache_dir / "hubconf.py").exists():
                        LOGGER.info("hub.load(%s) -> local cache %s", repo, cache_dir.name)
                        return self._orig(str(cache_dir), model, *args, source="local", **kwargs)
            return self._orig(repo, model, *args, source=source, **kwargs)

        torch.hub.load = patched
        return self

    def __exit__(self, *exc) -> None:
        import torch

        if self._orig is not None:
            torch.hub.load = self._orig


def _seed_extra_hub_repos() -> None:
    for repo in _offline_hub._EXTRA_REPOS:
        _ensure_hub_repo(repo, "master")


def load_depth_model(profile: str = "auto", device: str = "auto"):
    """Load ZoeDepth or MiDaS per profile; returns (model, engine_name)."""
    import torch

    dev = utils.torch_device(device)
    want_zoe = profile in ("auto", "zoedepth", "paper")
    want_midas = profile in ("auto", "midas", "cpu")

    if want_zoe:
        try:
            cache_key = f"zoe_{profile}_{dev.type}"
            if cache_key in _DEPTH_MODEL_CACHE:
                return _DEPTH_MODEL_CACHE[cache_key], "zoedepth_nk"
            LOGGER.info("Loading ZoeD_NK from torch.hub ( isl-org/ZoeDepth ) ...")
            _ensure_hub_repo("isl-org/ZoeDepth", "main")
            _seed_extra_hub_repos()
            with _offline_hub():
                zoe = torch.hub.load(str(_hub_repo_dir("isl-org/ZoeDepth", "main")),
                                     "ZoeD_NK", source="local", pretrained=True,
                                     map_location=dev.type)
            zoe = zoe.to(dev).eval()
            _DEPTH_MODEL_CACHE.clear()
            _DEPTH_MODEL_CACHE[cache_key] = zoe
            return zoe, "zoedepth_nk"
        except Exception as exc:  # pragma: no cover - network / RAM dependent
            LOGGER.warning("ZoeDepth unavailable ( %s ) - falling back to MiDaS_small", exc)

    if want_midas or profile in ("auto", "midas", "cpu"):
        cache_key = f"midas_{dev.type}"
        if cache_key in _DEPTH_MODEL_CACHE:
            return _DEPTH_MODEL_CACHE[cache_key], "midas_small"
        LOGGER.info("Loading MiDaS_small from torch.hub ( intel-isl/MiDaS ) ...")
        _seed_extra_hub_repos()
        repo_dir = _hub_repo_dir("intel-isl/MiDaS", "master")
        with _offline_hub():
            midas = torch.hub.load(str(repo_dir), "MiDaS_small", source="local")
        midas = midas.to(dev).eval()
        transforms = torch.hub.load(str(repo_dir), "transforms", source="local")
        small_transform = transforms.small_transform
        bundle = (midas, small_transform)
        _DEPTH_MODEL_CACHE.clear()
        _DEPTH_MODEL_CACHE[cache_key] = bundle
        return bundle, "midas_small"

    raise ValueError(f"Unknown depth profile: {profile}")


def release_models() -> None:
    _DEPTH_MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict(model, engine: str, image_rgb: np.ndarray, device: str = "auto") -> np.ndarray:
    """Run the depth model; returns raw ( H, W ) float map, 1 = nearest."""
    import torch

    dev = utils.torch_device(device)
    h, w = image_rgb.shape[:2]
    pil = Image.fromarray(np.ascontiguousarray(image_rgb))

    with torch.inference_mode():
        if engine == "zoedepth_nk":
            metric = model.infer_pil(pil)  # numpy, larger = farther ( meters )
            metric = np.asarray(metric, dtype=np.float32)
            if metric.shape != (h, w):
                pil_metric = Image.fromarray(metric.astype(np.float32), mode="F").resize(
                    (w, h), resample=Image.BILINEAR)
                metric = np.asarray(pil_metric, dtype=np.float32)
            near = 1.0 - utils.unit01(metric)  # invert: 1 = nearest
            return near.astype(np.float32)

        # MiDaS small bundle ( hub transforms take numpy arrays, not PIL )
        midas, transform = model
        tensor = transform(np.ascontiguousarray(image_rgb)).to(dev)
        prediction = midas(tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
        relative = prediction.squeeze().cpu().numpy().astype(np.float32)
        return utils.percentile_normalize(relative)  # MiDaS: larger = closer


def smooth_depth(depth01: np.ndarray, k: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Edge-preserving-ish smoothing: median kills speckle, gaussian softens
    banding so the parallax warp stays stable in the renderer."""
    import cv2

    out = cv2.medianBlur(depth01.astype(np.float32), k)
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def depth_colormap(depth01: np.ndarray) -> np.ndarray:
    """Turbo colormap visualisation ( near = warm ) for quick inspection."""
    import cv2

    u8 = (np.clip(depth01, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_depth(image, out_dir: Path | str, profile: str = "auto", device: str = "auto",
              force: bool = False) -> dict:
    """Estimate depth for ``image`` and persist it under ``out_dir/depth``.

    Returns the manifest fragment describing the depth assets, including the
    per-layer median depth ( merged with ``layers.json`` downstream ).
    """
    if isinstance(image, (str, Path)):
        image_rgb = utils.load_image_rgb(image)
    else:
        image_rgb = np.asarray(image)

    out_dir = ensure_dir(Path(out_dir))
    depth_dir = ensure_dir(out_dir / "depth")
    json_path = depth_dir / "depth.json"
    if json_path.exists() and not force:
        LOGGER.info("depth.json exists - skipping depth stage ( use --force to rerun )")
        return utils.read_json(json_path)

    model, engine = load_depth_model(profile, device)
    with utils.timed(f"Depth inference ({engine})"):
        near = _predict(model, engine, image_rgb, device)
    release_models()

    with utils.timed("Depth post-processing"):
        smooth = smooth_depth(near)
        np.save(depth_dir / "depth.npy", near.astype(np.float32))
        np.save(depth_dir / "depth_smooth.npy", smooth)
        gray = (np.clip(smooth, 0, 1) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(gray).save(depth_dir / "depth.png")
        Image.fromarray(depth_colormap(smooth)).save(depth_dir / "depth_viz.png")

        # per-layer median depth ( parallax amplitude per paint layer )
        layer_depth: Dict[str, float] = {}
        layers_json = out_dir / "layers.json"
        if layers_json.exists():
            for entry in utils.read_json(layers_json):
                name = entry["name"]
                mask_path = out_dir / "layers" / f"{name}.png"
                if not mask_path.exists():
                    continue
                alpha = np.asarray(Image.open(mask_path).convert("RGBA"))[..., 3]
                m = alpha > 127
                if m.any():
                    layer_depth[name] = round(float(np.median(smooth[m])), 4)

    manifest = {
        "engine": engine,
        "file": "depth/depth.png",
        "rawFile": "depth/depth.npy",
        "smoothFile": "depth/depth_smooth.npy",
        "vizFile": "depth/depth_viz.png",
        "bitDepth": 8,
        "nearIsOne": True,
        "layerDepth": layer_depth,
        "stats": {
            "min": round(float(smooth.min()), 4),
            "max": round(float(smooth.max()), 4),
            "mean": round(float(smooth.mean()), 4),
        },
    }
    atomic_write_json(json_path, manifest)
    LOGGER.info("Depth written to %s ( engine: %s )", depth_dir, engine)
    return manifest
