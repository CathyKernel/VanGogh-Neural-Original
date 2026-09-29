"""
Generate segmentation layer masks with MobileSAM (vit_t) using point prompts.

Layers (semantic regions of Starry Night):
  0 sky, 1 moon, 2 stars, 3 cypress, 4 village, 5 mountains
Priority: moon > stars > cypress > village > mountains; sky = complement.

Outputs to /home/z/my-project/cv_work/assets/layers/
"""
import os, sys, json, time
import numpy as np
import cv2
import torch

T0 = time.time()
SRC = "/home/z/my-project/upload/VanGogh_extracted/original/starry-night.jpg"
OUT_DIR = "/home/z/my-project/cv_work/assets/layers"
MS_CKPT = "/home/z/my-project/cv_work/mobile_sam.pt"
MS_REPO = "/home/z/my-project/cv_work/MobileSAM"

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- image + luminous center detection (same as flow script) ----------------
img_bgr = cv2.imread(SRC, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, (18, 70, 140), (48, 255, 255))
n, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
blobs = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    if area >= 250 and 15 < cx < W - 15 and 15 < cy < H - 15:
        blobs.append({"cx": cx, "cy": cy, "area": area})
blobs.sort(key=lambda c: -c["area"])
moon_pt = np.array([[blobs[0]["cx"], blobs[0]["cy"]]])
star_pts = np.array([[b["cx"], b["cy"]] for b in blobs[1:]])
log(f"image {W}x{H}; {len(blobs)} interior luminous blobs "
    f"(moon + {len(star_pts)} stars)")

# ---------------- MobileSAM ----------------
sys.path.insert(0, MS_REPO)
from mobile_sam import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_t"](checkpoint=MS_CKPT)
sam.eval()
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)
log("MobileSAM image embedding computed")

def predict_mask(points, labels_):
    m, s, _ = predictor.predict(point_coords=np.array(points), point_labels=np.array(labels_),
                                multimask_output=False)
    return m[0], float(s[0])

def predict_compact(pts, max_frac=0.025, radius=70):
    """Predict each point separately, keep only the connected component containing
    the prompt point (within `radius`), capped at max_frac of the image. Union."""
    out = np.zeros((H, W), dtype=bool)
    for (px, py) in pts:
        m, s = predict_mask([[px, py]], [1])
        m = m.astype(np.uint8)
        nl, ll, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, nl):
            bx, by, bw, bh, area = st[i]
            # component bbox must intersect the prompt neighborhood
            near = not (bx > px + radius or bx + bw < px - radius or
                        by > py + radius or by + bh < py - radius)
            if near and area <= max_frac * W * H:
                out |= (ll == i)
    return out

def clean(mask):
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)

# ---------------- per-layer prompts ----------------
moon_mask, s_moon = predict_mask(moon_pt, [1])
log(f"moon mask: score={s_moon:.3f} coverage={moon_mask.mean()*100:.1f}%")

stars_mask = predict_compact(star_pts)
s_star = float("nan")  # per-star union has no single score
log(f"stars mask (per-star union): coverage={stars_mask.mean()*100:.1f}%")

cyp_pos = [[0.175 * W, 0.32 * H], [0.155 * W, 0.55 * H], [0.21 * W, 0.78 * H]]
cyp_neg = [[0.45 * W, 0.30 * H], [0.05 * W, 0.05 * H]]
cypress_mask, s_cyp = predict_mask(cyp_pos + cyp_neg, [1, 1, 1, 0, 0])
log(f"cypress mask: score={s_cyp:.3f} coverage={cypress_mask.mean()*100:.1f}%")

vil_pos = [[0.40 * W, 0.92 * H], [0.52 * W, 0.93 * H], [0.62 * W, 0.95 * H]]
vil_neg = [[0.35 * W, 0.78 * H], [0.55 * W, 0.76 * H], [0.75 * W, 0.72 * H], [0.85 * W, 0.60 * H]]
village_mask, s_vil = predict_mask(vil_pos + vil_neg, [1, 1, 1, 0, 0, 0, 0])
# geometric constraint: village sits in the bottom band of the painting
band = np.zeros((H, W), dtype=bool); band[int(0.74 * H):, :] = True
village_mask = village_mask & band
log(f"village mask: score={s_vil:.3f} coverage={village_mask.mean()*100:.1f}% (bottom-band constrained)")

mtn_pos = [[0.12 * W, 0.78 * H], [0.48 * W, 0.75 * H], [0.85 * W, 0.73 * H]]
mtn_neg = [[0.30 * W, 0.55 * H], [0.70 * W, 0.50 * H], [0.45 * W, 0.92 * H]]
mountain_mask, s_mtn = predict_mask(mtn_pos + mtn_neg, [1, 1, 1, 0, 0, 0])
# geometric constraint: the Alpilles ridge is a horizontal band
band = np.zeros((H, W), dtype=bool); band[int(0.64 * H):int(0.88 * H), :] = True
mountain_mask = mountain_mask & band
log(f"mountains mask: score={s_mtn:.3f} coverage={mountain_mask.mean()*100:.1f}% (ridge-band constrained)")

# ---------------- priority assignment + sky complement ----------------
layers = [
    ("moon", clean(moon_mask)),
    ("stars", clean(stars_mask)),
    ("cypress", clean(cypress_mask)),
    ("village", clean(village_mask)),
    ("mountains", clean(mountain_mask)),
]
assigned = np.zeros((H, W), dtype=bool)
final = {}
for name, m in layers:                    # earlier layers win
    m = m & ~assigned
    assigned |= m
    final[name] = m
sky = ~assigned                           # everything else = sky
final["sky"] = sky
order = ["sky", "moon", "stars", "cypress", "village", "mountains"]

# ---------------- save masks + metadata ----------------
meta = {"model": "MobileSAM (vit_t, TinyViT image encoder)",
        "prompting": "point prompts (positive=forground, 0=background), single mask output",
        "priority": ["moon", "stars", "cypress", "village", "mountains"],
        "sky": "complement of all other layers",
        "source": {"width": W, "height": H},
        "prompt_scores": {"moon": round(s_moon, 3), "stars": round(s_star, 3),
                          "cypress": round(s_cyp, 3), "village": round(s_vil, 3),
                          "mountains": round(s_mtn, 3)},
        "files": [], "layers": []}
COLORS = {"sky": (90, 60, 30), "moon": (60, 220, 255), "stars": (80, 200, 255),
          "cypress": (70, 90, 40), "village": (160, 90, 60), "mountains": (150, 110, 70)}  # BGR

for i, name in enumerate(order):
    m = final[name]
    cov = float(m.mean())
    ys, xs = np.where(m)
    cx = float(xs.mean()) if len(xs) else 0.0
    cy = float(ys.mean()) if len(ys) else 0.0
    mean_col = img_bgr[m].mean(axis=0) if m.any() else [0, 0, 0]
    fname = f"layer_{i}_{name}.png"
    cv2.imwrite(f"{OUT_DIR}/{fname}", (m * 255).astype(np.uint8))
    meta["files"].append(fname)
    meta["layers"].append({
        "id": i, "name": name, "file": fname, "color_bgr": [int(c) for c in COLORS[name]],
        "coverage_pct": round(cov * 100, 2), "centroid": [round(cx, 1), round(cy, 1)],
        "mean_color_bgr": [round(float(c), 1) for c in mean_col],
    })
    log(f"saved {fname}: coverage={cov*100:.2f}% mean_bgr={[round(float(c)) for c in mean_col]}")

# ---------------- preview overlay ----------------
overlay = img_bgr.copy()
for i, name in enumerate(order):
    tinted = img_bgr.copy()
    tinted[:] = COLORS[name]
    overlay = np.where(final[name][..., None], (0.55 * tinted + 0.45 * overlay), overlay)
preview = np.clip(overlay * 0.75 + 30, 0, 255).astype(np.uint8)
for i, name in enumerate(order):
    L = meta["layers"][i]
    x, y = int(L["centroid"][0]), int(L["centroid"][1])
    cv2.putText(preview, f"{i}:{name}", (max(10, x - 60), max(24, y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(preview, (x, y), 5, (255, 255, 255), -1)
cv2.imwrite(f"{OUT_DIR}/layers_preview.png", preview)
meta["files"].append("layers_preview.png")
log("layers_preview.png written")

meta["total_coverage_pct"] = round(sum(l["coverage_pct"] for l in meta["layers"]), 2)
meta["seconds"] = round(time.time() - T0, 1)
with open(f"{OUT_DIR}/layers_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
log("layers_meta.json written")
print("LAYERS GENERATION COMPLETE")
