"""Depth asset — MiDaS v2.1 small (efficientnet_lite3 backbone).

Citation:
  Ranftl, Lasinger, Hafner, Schindler, Ranftl,
  "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot
  Cross-dataset Transfer", IEEE TPAMI 2022.

The model estimates relative inverse depth from a single RGB image
(larger value = nearer). The WebGL player uses it for pointer parallax:
nearer regions (the village band) shift more than far ones (the moon).

Run from anywhere inside the repository:
    python cv_pipeline/gen_depth.py

Writes cv_pipeline/assets/depth/depth_midas_small.png
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

ROOT = Path(__file__).resolve().parents[1]          # repository root
OUT = Path(__file__).resolve().parent / 'assets/depth'
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(ROOT / 'starry-night.jpg'), cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# official entry point; downloads the checkpoint on first run (~82 MB)
midas = torch.hub.load('isl-org/MiDaS', 'MiDaS_small')
midas.eval()
transform = torch.hub.load('isl-org/MiDaS', 'transforms').small_transform

with torch.no_grad():
    x = transform({'image': img_rgb})
    if isinstance(x, dict):
        x = x['image']
    pred = midas(x)
    pred = F.interpolate(pred.unsqueeze(1), size=(H, W), mode='bicubic',
                         align_corners=False).squeeze().numpy()

d = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
cv2.imwrite(str(OUT / 'depth_midas_small.png'), (d * 255.0).astype(np.uint8))

with open(OUT / 'depth_stats.json', 'w') as f:
    json.dump({'model': 'MiDaS v2.1 small (efficientnet_lite3, 256px input)',
              'source': {'width': W, 'height': H},
              'convention': '255 = nearest (village), 0 = farthest (moon sky)',
              'raw_min': float(pred.min()), 'raw_max': float(pred.max())}, f, indent=2)
print(f'wrote {OUT / "depth_midas_small.png"}')
