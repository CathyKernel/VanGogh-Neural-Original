"""RAFT 16-phase periodic flow cycle — the core neural motion asset (v5).

Citations:
  Teed, Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow",
  ECCV 2020 (Oral, Best Paper Award). torchvision official implementation,
  raft_small with C+T (V2) weights.
  Schodl, Szeliski, Salesin, Essa, "Video Textures", SIGGRAPH 2000 — the
  periodic-loop construction that makes the cycle seamlessly infinite.

Method (all PyTorch, no hand-drawn animation in the final asset):
  1. A region-weighted periodic motion model M(phi), phi in [0,1), describes
     how each SAM layer should move (oscillating vortices over the two great
     sky swirls and the moon halo, a band current through the sky, cypress
     sway, village / mountain drift, global breathing). All temporal terms
     are integer harmonics of phi, so the cycle is exactly loopable.
  2. Sixteen phase frames are synthesized by warping the ORIGINAL painting
     with torch.nn.functional.grid_sample:
         F_k(p) = original(p + d_k(p)),   d_k = M(phi_k) - M(phi_0)
     so F_0 is the untouched original.
  3. RAFT-small MEASURES the correspondence of every phase back to the
     original, B_k = RAFT(F_k, F_0). B_k(p) is exactly the field the WebGL
     player needs: where pixel p of phase k samples the original from.
  4. The 16 fields are encoded into a 4x4 PNG atlas (tile k = phase k,
     flow_px = (texel.rg - 0.5) * 64.0).

Run from anywhere inside the repository (after gen_layers.py):
    python cv_pipeline/gen_flow.py

Writes cv_pipeline/assets/flow/flow_raft_phases.png (+ stats + visualization)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'assets/flow'
OUT.mkdir(parents=True, exist_ok=True)

N_PHASES = 16
TILE_W, TILE_H = 320, 254
FLOW_RANGE = 32.0          # encoded px range: (texel - 0.5) * 2 * FLOW_RANGE
MAX_DISP = 14.0            # motion model cap, px
RW, RH = 640, 504          # RAFT input size (must be divisible by 8)

torch.set_grad_enabled(False)

def load_rgb(path):
    arr = np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def load_mask(path, blur=9):
    arr = np.asarray(Image.open(path).convert('L')).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    k = torch.arange(-blur, blur + 1, dtype=torch.float32)
    g = torch.exp(-(k ** 2) / (2.0 * (blur / 2.0) ** 2))
    g = g / g.sum()
    t = TF.conv2d(TF.pad(t, (blur, blur, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
    t = TF.conv2d(TF.pad(t, (0, 0, blur, blur), mode='replicate'), g.view(1, 1, -1, 1))
    return t[0, 0]

orig = load_rgb(ROOT / 'starry-night.jpg')
_, _, H, W = orig.shape
LAY = Path(__file__).resolve().parent / 'assets/layers'
layers = {k: load_mask(LAY / f) for k, f in [
    ('sky', 'layer_0_sky.png'), ('moon', 'layer_1_moon.png'),
    ('stars', 'layer_2_stars.png'), ('cypress', 'layer_3_cypress.png'),
    ('village', 'layer_4_village.png'), ('mountains', 'layer_5_mountains.png')]}
sky_w = layers['sky'] + layers['moon'] + layers['stars']

Y, X = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                      torch.arange(W, dtype=torch.float32), indexing='ij')
TAU = 2.0 * math.pi

VORTICES = [   # (cx, cy, sigma_px, theta_amplitude_rad, phase_offset)
    (461.0, 342.0, 300.0, 0.045, 0.0),    # grand swirl A (upper-left sky)
    (818.0, 456.0, 260.0, 0.050, 2.1),    # grand swirl B (center-right sky)
    (1143.0, 179.0, 130.0, 0.060, 4.0),   # moon halo swirl
]

def motion_field(phi):
    dx = torch.zeros(H, W)
    dy = torch.zeros(H, W)
    for cx, cy, sig, th_amp, ph in VORTICES:
        rx, ry = X - cx, Y - cy
        w = torch.exp(-(rx * rx + ry * ry) / (2.0 * sig * sig))
        th = th_amp * math.sin(TAU * phi + ph)
        c, s = math.cos(th), math.sin(th)
        dx = dx + sky_w * w * ((c - 1.0) * rx - s * ry)
        dy = dy + sky_w * w * (s * rx + (c - 1.0) * ry)
    dx = dx + sky_w * 5.5 * torch.sin(TAU * phi - Y / 170.0 + 0.5)
    dy = dy + sky_w * 1.8 * torch.sin(TAU * phi + X / 240.0 + 1.2)
    hfac = torch.clamp(1.0 - Y / H, 0.0, 1.0)
    dx = dx + layers['cypress'] * (2.0 + 3.0 * hfac) * torch.sin(TAU * phi + hfac * 2.6 + 0.9)
    dy = dy + layers['cypress'] * 1.2 * torch.sin(TAU * phi + X / 150.0 + 2.4)
    dx = dx + layers['village'] * 2.6 * math.sin(TAU * phi + 0.8)
    dy = dy + layers['village'] * 1.0 * math.sin(2.0 * TAU * phi + 2.9)
    dx = dx + layers['mountains'] * 2.2 * math.sin(TAU * phi + 2.1)
    dy = dy + layers['mountains'] * 0.8 * math.sin(TAU * phi + 3.7)
    s = 0.006 * math.sin(TAU * phi + 1.3)
    dx = dx + (X - W / 2.0) * s
    dy = dy + (Y - H / 2.0) * s
    mag = torch.sqrt(dx * dx + dy * dy)
    scale = torch.clamp(MAX_DISP / mag, max=1.0)
    return dx * scale, dy * scale

def warp_original(dx, dy):
    sx = (X + dx) / (W - 1) * 2 - 1
    sy = (Y + dy) / (H - 1) * 2 - 1
    grid = torch.stack([sx, sy], dim=-1).unsqueeze(0)
    return TF.grid_sample(orig, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)

M0 = motion_field(0.0)
raft = raft_small(weights=Raft_Small_Weights.C_T_V2).eval()
SX, SY = W / RW, H / RH

atlas = torch.zeros(4 * TILE_H, 4 * TILE_W, 3)
stats = {'phases': [], 'regions': {}}
t_start = time.time()
phase_frames, phase_flows = [], []

for k in range(N_PHASES):
    dxk, dyk = motion_field(k / N_PHASES)
    frame = warp_original(dxk - M0[0], dyk - M0[1])
    phase_frames.append(frame)

    fA = TF.interpolate(frame, size=(RH, RW), mode='bilinear', align_corners=True)
    fB = TF.interpolate(phase_frames[0], size=(RH, RW), mode='bilinear', align_corners=True)
    with torch.inference_mode():
        flow = raft(fA, fB)[-1]
    flow = TF.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
    Bk = torch.stack([flow[0, 0] * SX, flow[0, 1] * SY], dim=-1)   # H,W,2 px
    phase_flows.append(Bk.to(torch.float16))

    bxh = TF.interpolate(Bk[..., 0].unsqueeze(0).unsqueeze(0), size=(TILE_H, TILE_W),
                         mode='bilinear', align_corners=True)[0, 0]
    byh = TF.interpolate(Bk[..., 1].unsqueeze(0).unsqueeze(0), size=(TILE_H, TILE_W),
                         mode='bilinear', align_corners=True)[0, 0]
    r = torch.clamp(bxh / (2 * FLOW_RANGE) + 0.5, 0.0, 1.0)
    g = torch.clamp(byh / (2 * FLOW_RANGE) + 0.5, 0.0, 1.0)
    ty, tx = divmod(k, 4)
    atlas[ty * TILE_H:(ty + 1) * TILE_H, tx * TILE_W:(tx + 1) * TILE_W] = \
        torch.stack([r, g, torch.zeros_like(r)], dim=-1)

    mag = torch.sqrt(Bk[..., 0] ** 2 + Bk[..., 1] ** 2)
    stats['phases'].append({'k': k, 'mean_px': round(float(mag.mean()), 3),
                           'max_px': round(float(mag.max()), 2),
                           'pct_gt_1px': round(float((mag > 1.0).float().mean() * 100), 1)})
    print(f"phase {k:2d}: mean={stats['phases'][-1]['mean_px']:5.2f}px  "
          f">1px {stats['phases'][-1]['pct_gt_1px']:5.1f}%")

mag4 = torch.sqrt(phase_flows[4][..., 0].float() ** 2 + phase_flows[4][..., 1].float() ** 2)
for key, m in layers.items():
    stats['regions'][key] = round(float((mag4 * m).sum() / max(float(m.sum()), 1)), 3)

Image.fromarray((atlas.numpy() * 255.0).round().astype(np.uint8)).save(OUT / 'flow_raft_phases.png')
mn, mx = float(mag4.min()), float(mag4.max())
Image.fromarray(((mag4 - mn) / (mx - mn + 1e-6) * 255).numpy().astype(np.uint8)).save(
    OUT / 'flow_mag_phase4.png')
stats['model'] = {'flow': 'RAFT-small (torchvision official, C+T v2 weights)',
                 'synthesis': 'torch.nn.functional.grid_sample',
                 'n_phases': N_PHASES, 'atlas': '4x4, flow_px=(texel.rg-0.5)*64',
                 'seconds': round(time.time() - t_start, 1)}
with open(OUT / 'flow_cycle_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"wrote {OUT / 'flow_raft_phases.png'} in {stats['model']['seconds']}s")
