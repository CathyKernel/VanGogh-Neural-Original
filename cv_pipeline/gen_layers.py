"""Segmentation layers — MobileSAM (vit_t TinyViT image encoder).

Citations:
  Kirillov et al., "Segment Anything", ICCV 2023.
  Zhang et al., "MobileSAM: Faster Segment Anything", 2023
  (vit_t TinyViT encoder, ~40 MB).

Six semantic layers of The Starry Night, produced with point prompts
(moon / stars located by an HSV luminous-blob detector, cypress / village /
mountains by manually chosen points, sky as the complement). Each star is
predicted independently and filtered by connected-component size so that
brushstroke-consistent sky is never absorbed into a single star mask.

Run from anywhere inside the repository (requires the MobileSAM package and
checkpoint, see cv_pipeline/README.md):
    python cv_pipeline/gen_layers.py

Writes cv_pipeline/assets/layers/layer_0..5_*.png
"""
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'assets/layers'
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / 'mobile_sam.pt'                      # download: see cv_pipeline/README.md

img_bgr = cv2.imread(str(ROOT / 'starry-night.jpg'), cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---- luminous centers (moon + stars) via HSV thresholding ----
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, (18, 70, 140), (48, 255, 255))
n, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
blobs = [{'cx': c[0], 'cy': c[1], 'area': s[4]} for i, (c, s) in
         enumerate(zip(centroids, stats)) if i > 0 and s[4] >= 250]
blobs.sort(key=lambda b: -b['area'])
moon_pt = (blobs[0]['cx'], blobs[0]['cy'])
star_pts = [(b['cx'], b['cy']) for b in blobs[1:]]

# ---- MobileSAM ----
sam = sam_model_registry['vit_t'](checkpoint=str(CKPT))
sam.eval()
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

def predict(points):
    m, s, _ = predictor.predict(point_coords=np.array(points),
                                point_labels=[1] * len(points),
                                multimask_output=False)
    return m[0].astype(bool)

def compact(pts, max_frac=0.025, radius=70):
    """Predict each point separately; keep the connected component that contains
    the point if it is smaller than max_frac of the image (prevents one star from
    absorbing the whole sky, which SAM happily does)."""
    acc = np.zeros((H, W), bool)
    for (px, py) in pts:
        m = predict([(px, py)])
        if m.mean() > max_frac:
            continue
        mm, ll, ss, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, mm):
            cx, cy = ss[i][0] + ss[i][2] / 2, ss[i][1] + ss[i][3] / 2
            if (cx - px) ** 2 + (cy - py) ** 2 < radius ** 2:
                acc |= ll == i
                break
    return acc

layers = {
    'moon':      compact([moon_pt], max_frac=0.05),
    'stars':     compact(star_pts),
    'cypress':   predict([(205, 320), (165, 560), (255, 700)]),
    'village':   predict([(430, 850), (620, 880), (830, 900)]),
    'mountains': predict([(1050, 730), (1150, 690)]),
}
# spatial sanity gates (the painting's layout)
yy = np.arange(H).reshape(-1, 1)
layers['village'] &= yy > 0.74 * H
layers['mountains'] &= (yy > 0.64 * H) & (yy < 0.88 * H)

covered = layers['moon'] | layers['stars'] | layers['cypress'] | layers['village'] | layers['mountains']
layers['sky'] = ~covered

names = ['sky', 'moon', 'stars', 'cypress', 'village', 'mountains']
meta = {}
for i, name in enumerate(names):
    m = layers[name].astype(np.uint8) * 255
    cv2.imwrite(str(OUT / f'layer_{i}_{name}.png'), m)
    meta[name] = round(float(layers[name].mean() * 100), 2)
with open(OUT / 'layers_meta.json', 'w') as f:
    json.dump({'model': 'MobileSAM vit_t (TinyViT encoder)', 'layers_pct': meta}, f, indent=2)
print(f'wrote 6 layer masks: {meta}')
