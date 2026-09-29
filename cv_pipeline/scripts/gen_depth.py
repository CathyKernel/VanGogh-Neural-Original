"""
Generate depth map assets with MiDaS v2.1 small (efficientnet_lite3 backbone).
Outputs to /home/z/my-project/cv_work/assets/depth/
  - depth_midas_small.png       8-bit grayscale PNG (255 = nearest)
  - depth_midas_small_16bit.png 16-bit PNG (65535 = nearest)
  - depth_stats.json            statistics for manifest
"""
import sys, json, time, importlib.util
import numpy as np
import cv2
import torch

T0 = time.time()
SRC = "/home/z/my-project/cv_work/starry-night.jpg"
OUT_DIR = "/home/z/my-project/cv_work/assets/depth"
MIDAS_CKPT = "/home/z/my-project/cv_work/midas_v21_small.pt"
MIDAS_REPO = "/home/z/my-project/cv_work/midas_repo"

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# --- patch torch.hub fork-validation bug (KeyError 'Authorization' on API rate limit) ---
torch.hub._validate_not_a_forked_repo = lambda *a, **k: None

# --- load MiDaS small from local repo clone, weights from local checkpoint ---
sys.path.insert(0, MIDAS_REPO)
spec = importlib.util.spec_from_file_location("midas_hub", MIDAS_REPO + "/hubconf.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.MiDaS_small(pretrained=False)
sd = torch.load(MIDAS_CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.eval()
log("MiDaS small model ready")

# --- MiDaS small_transform (replicated from hubconf.transforms().small_transform) ---
from torchvision.transforms import Compose
from midas.transforms import Resize, NormalizeImage, PrepareForNet
transform = Compose([
    lambda img: {"image": img / 255.0},
    Resize(256, 256, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
           resize_method="upper_bound", image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
])

# --- load source image (RGB float) ---
img_bgr = cv2.imread(SRC, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
log(f"source image {W}x{H}")

with torch.no_grad():
    x = transform(img_rgb)
    if isinstance(x, dict):
        x = x["image"]
    log(f"transformed input {tuple(x.shape)}")
    pred = model(x)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    ).squeeze().numpy()
log(f"inference done, raw range [{pred.min():.4f}, {pred.max():.4f}]")

d = pred - pred.min()
d = d / (d.max() + 1e-8)
depth8 = (d * 255.0).astype(np.uint8)
depth16 = (d * 65535.0).astype(np.uint16)

import os
os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(f"{OUT_DIR}/depth_midas_small.png", depth8)
cv2.imwrite(f"{OUT_DIR}/depth_midas_small_16bit.png", depth16)

stats = {
    "model": "MiDaS v2.1 small (efficientnet_lite3 backbone, lightweight 256px)",
    "checkpoint": "midas_v21_small.pt (v2_1 release)",
    "source": {"width": W, "height": H},
    "convention": "larger value = nearer to camera (relative inverse depth); the model treats the village band as nearest and the moon/stars as farthest",
    "raw_min": float(pred.min()), "raw_max": float(pred.max()),
    "raw_mean": float(pred.mean()),
    "seconds": round(time.time() - T0, 1),
}
with open(f"{OUT_DIR}/depth_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
log(f"saved depth_midas_small.png / depth_midas_small_16bit.png -> {OUT_DIR}")
print("DEPTH GENERATION COMPLETE")
