"""
Generate optical flow assets with RAFT-small (torchvision official implementation).

Method:
  1. Auto-detect luminous centers (moon + stars) via HSV color thresholding.
  2. Build a synthetic Van-Gogh-style swirl displacement field around each center.
  3. Warp the painting by that field -> synthetic "next frame".
  4. Run REAL RAFT-small optical flow estimation between original and warped frame.
  5. Save Middlebury .flo + RGB-encoded PNG + visualization + stats.

Outputs to /home/z/my-project/cv_work/assets/flow/
"""
import os, json, time
import numpy as np
import cv2
import torch

T0 = time.time()
SRC = "/home/z/my-project/upload/VanGogh_extracted/original/starry-night.jpg"
OUT_DIR = "/home/z/my-project/cv_work/assets/flow"
FLOW_SCALE_PX = 16.0   # encoding scale for the RGB PNG

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 1. load image ----------------
img_bgr = cv2.imread(SRC, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
log(f"source {W}x{H}")

# ---------------- 2. detect luminous centers (moon + stars) ----------------
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, (18, 70, 140), (48, 255, 255))
n, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=8)
centers = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area >= 250:
        centers.append({"cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
                        "area": float(area), "bbox": [int(x), int(y), int(w), int(h)]})
centers.sort(key=lambda c: -c["area"])
centers[0]["role"] = "moon"
for c in centers[1:]:
    c["role"] = "star"
log(f"detected {len(centers)} luminous centers: "
    f"moon area={centers[0]['area']:.0f}px, then {len(centers)-1} star/halo blobs")
for c in centers:
    c["sigma"] = float(np.sqrt(c["area"]) * 1.2)
    c["strength"] = float(min(3.0 + np.sqrt(c["area"]) / 22.0, 11.0))
    c["dir"] = 1.0 if centers.index(c) % 2 == 0 else -1.0

# ---------------- 3. swirl displacement field ----------------
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
dx = np.zeros((H, W), dtype=np.float32)
dy = np.zeros((H, W), dtype=np.float32)
for c in centers:
    rx = gx - c["cx"]; ry = gy - c["cy"]
    r2 = rx * rx + ry * ry
    fall = np.exp(-r2 / (2.0 * c["sigma"] ** 2))
    # tangential (perpendicular) direction = (-ry, rx) normalized
    r = np.sqrt(r2) + 1e-3
    tx = -ry / r; ty = rx / r
    dx += c["dir"] * c["strength"] * fall * tx
    dy += c["dir"] * c["strength"] * fall * ty
mag = np.sqrt(dx * dx + dy * dy)
scale = np.minimum(1.0, 14.0 / (mag + 1e-6))
dx *= scale; dy *= scale
log(f"swirl field: mean|d|={np.sqrt(dx*dx+dy*dy).mean():.2f}px max={np.sqrt(dx*dx+dy*dy).max():.2f}px")

# ---------------- 4. warped "next frame" ----------------
map_x = (gx - dx).astype(np.float32)   # frame2(q) = frame1(q - d(q))
map_y = (gy - dy).astype(np.float32)
frame2 = cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
cv2.imwrite("/home/z/my-project/cv_work/frame2_swirl.jpg", frame2)
log("synthetic next frame (swirl-warped) written")

# ---------------- 5. RAFT-small inference ----------------
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).eval()

def to_tensor(im_bgr, size):
    im = cv2.resize(im_bgr, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
    return (t - 0.5) / 0.5   # raft weights normalize to [-1, 1]

RW, RH = 960, 760            # both divisible by 8, aspect ~= source (1.263 vs 1.262)
t1 = to_tensor(img_bgr, (RW, RH))
t2 = to_tensor(frame2, (RW, RH))
with torch.no_grad():
    flows = model(t1, t2, num_flow_updates=12)
flow = flows[-1][0].numpy()  # (2, RH, RW), pixels in RWxRH space
log(f"RAFT done: mean|f|={np.linalg.norm(flow, axis=0).mean():.2f}px max={np.linalg.norm(flow, axis=0).max():.2f}px")

# ---------------- 6. upsample to full resolution, rescale values ----------------
flow = flow.transpose(1, 2, 0)  # (RH, RW, 2)
flow_full = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
flow_full[:, :, 0] *= W / RW   # u scales with x
flow_full[:, :, 1] *= H / RH   # v scales with y
log(f"upsampled to {W}x{H}: mean|f|={np.linalg.norm(flow_full, axis=2).mean():.2f}px "
    f"max={np.linalg.norm(flow_full, axis=2).max():.2f}px")

# ---------------- 7. save .flo (Middlebury format) ----------------
with open(f"{OUT_DIR}/flow_raft_small.flo", "wb") as f:
    f.write(np.array([202021.25], dtype=np.float32).tobytes())   # "PIEH" magic
    f.write(np.array([W, H], dtype=np.int32).tobytes())
    f.write(flow_full.astype(np.float32).tobytes())
log("flow_raft_small.flo written (Middlebury float32)")

# ---------------- 8. save RGB-encoded PNG (for direct GPU texture use) ----------------
u = np.clip(flow_full[:, :, 0] / (2 * FLOW_SCALE_PX) + 0.5, 0, 1)
v = np.clip(flow_full[:, :, 1] / (2 * FLOW_SCALE_PX) + 0.5, 0, 1)
m = np.clip(np.linalg.norm(flow_full, axis=2) / FLOW_SCALE_PX, 0, 1)
rgb = (np.stack([u, v, m], axis=-1) * 255.0).astype(np.uint8)
cv2.imwrite(f"{OUT_DIR}/flow_raft_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
log("flow_raft_rgb.png written (R=u, G=v, B=|f|; 128=zero flow)")

# ---------------- 9. save human visualization ----------------
viz = flow_to_image(torch.from_numpy(flow_full.transpose(2, 0, 1)))   # (3, H, W)
viz_np = np.asarray(viz).transpose(1, 2, 0)                            # (H, W, 3) RGB
cv2.imwrite(f"{OUT_DIR}/flow_raft_visualization.png", cv2.cvtColor(viz_np, cv2.COLOR_RGB2BGR))
log("flow_raft_visualization.png written")

# ---------------- 10. stats ----------------
magf = np.linalg.norm(flow_full, axis=2)
stats = {
    "model": "RAFT-small (torchvision official implementation, C+T weights)",
    "weights": "raft_small_C_T_V2",
    "method": ("RAFT flow between the original painting and a synthetic next frame "
               "warped by vortex displacement fields centered on auto-detected luminous "
               "elements (moon + stars)"),
    "source": {"width": W, "height": H},
    "raft_input": {"width": RW, "height": RH, "num_flow_updates": 12},
    "luminous_centers": [
        {"role": c["role"], "cx": round(c["cx"], 1), "cy": round(c["cy"], 1),
         "area_px": round(c["area"]), "swirl_strength_px": round(c["strength"], 2),
         "swirl_sigma_px": round(c["sigma"], 1), "direction": "+" if c["dir"] > 0 else "-"}
        for c in centers],
    "encoding": {
        "flo": "Middlebury float32, u then v per pixel, full 1280x1014",
        "rgb_png": f"R = u/({2*FLOW_SCALE_PX:.0f})+0.5, G = v/({2*FLOW_SCALE_PX:.0f})+0.5, "
                   f"B = |f|/{FLOW_SCALE_PX:.0f}; value 128 = zero flow, clamped",
        "scale_px": FLOW_SCALE_PX,
    },
    "flow_stats": {
        "mean_magnitude_px": round(float(magf.mean()), 3),
        "max_magnitude_px": round(float(magf.max()), 3),
        "pct_pixels_above_1px": round(float((magf > 1).mean() * 100), 1),
    },
    "seconds": round(time.time() - T0, 1),
}
with open(f"{OUT_DIR}/flow_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
log("flow_stats.json written")
print("FLOW GENERATION COMPLETE")
