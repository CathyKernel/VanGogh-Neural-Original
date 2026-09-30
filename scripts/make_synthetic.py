#!/usr/bin/env python3
"""Generate the deterministic synthetic Starry-Night-style test image.

A fully offline stand-in for Van Gogh's painting used by the pipeline demo
and the test-suite.  The generator paints, in classic CV terms:

* a deep-blue vertical gradient sky with two logarithmic-spiral swirl flows,
* eleven glowing stars with radial falloff and cross-flare,
* a crescent moon with halo,
* a mountain silhouette band,
* a village skyline with lit windows and a church spire,
* a flame-shaped cypress tree on the left.

Output: ``data/input/starry_night_synthetic.png`` ( 608 x 760, portrait,
same aspect as the original 73.7 x 92.1 cm canvas ).

Run:  python scripts/make_synthetic.py [--out path] [--seed 7]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 608, 760


def sky_gradient(rng: np.random.Generator) -> np.ndarray:
    """Deep blue vertical gradient with subtle turbulence."""
    top = np.array([10, 22, 66], np.float32)
    bottom = np.array([32, 66, 138], np.float32)
    t = np.linspace(0, 1, H)[:, None, None]
    field = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
    turb = rng.normal(0, 4.5, size=(H // 8, W // 8, 1)).repeat(8, 0).repeat(8, 1)
    field = np.clip(field + turb, 0, 255)
    return field.astype(np.float32)


def paint_swirls(field: np.ndarray, rng: np.random.Generator) -> None:
    """Two logarithmic spirals drawn as translucent brush arcs."""
    for cx, cy, hue, turns in ((int(W * 0.30), int(H * 0.28), (125, 168, 216), 1.35),
                               (int(W * 0.62), int(H * 0.16), (150, 186, 228), 1.6)):
        n = 260
        t = np.linspace(0.18, 1.0, n)
        theta = turns * 2 * np.pi * t + rng.uniform(0, 6.28)
        radius = 118 * (1.0 - 0.72 * t)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta) * 0.62
        color = np.array(hue, np.float32)
        for i in range(n - 1):
            alpha = 0.16 + 0.30 * t[i]
            thickness = max(2, int(9 * (1.0 - t[i]) + 1))
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            x_min, x_max = int(min(x0, x1)) - 2, int(max(x0, x1)) + 3
            y_min, y_max = int(min(y0, y1)) - 2, int(max(y0, y1)) + 3
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, W), min(y_max, H)
            if x_max <= x_min or y_max <= y_min:
                continue
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            d = np.abs((y1 - y0) * (xx - x0) - (x1 - x0) * (yy - y0)) / max(
                np.hypot(x1 - x0, y1 - y0), 1e-6)
            mask = d <= thickness / 2.0
            field[y_min:y_max, x_min:x_max][mask] = (
                field[y_min:y_max, x_min:x_max][mask] * (1 - alpha) + color * alpha)


def paint_glow(field: np.ndarray, x: float, y: float, radius: float,
               core: tuple, halo: tuple, flare: bool = False) -> None:
    x0, x1 = max(int(x - radius * 3), 0), min(int(x + radius * 3), W)
    y0, y1 = max(int(y - radius * 3), 0), min(int(y + radius * 3), H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    core_mask = np.clip(1 - d / radius, 0, 1) ** 0.8
    halo_mask = np.exp(-(d / (radius * 1.9)) ** 2) * 0.55
    region = field[y0:y1, x0:x1]
    for mask, color, gain in ((core_mask, core, 1.0), (halo_mask, halo, 1.0)):
        a = np.clip(mask * gain, 0, 1)[..., None]
        c = np.array(color, np.float32)
        region[:] = region * (1 - a) + c * a
    if flare:
        for angle_deg in (0, 90):
            ang = np.deg2rad(angle_deg)
            dx, dy = np.cos(ang), np.sin(ang)
            flare_d = np.abs((xx - x) * dy - (yy - y) * dx) / (radius * 0.30)
            flare_len = np.hypot(xx - x, yy - y) / (radius * 4.2)
            flare_mask = np.exp(-flare_d ** 2) * np.clip(1 - flare_len, 0, 1) ** 1.4 * 0.8
            a = flare_mask[..., None]
            c = np.array(core, np.float32)
            region[:] = region * (1 - a) + c * a


def paint_mountains(field: np.ndarray) -> None:
    xs = np.arange(W)
    ridge = (H * 0.615 - 36 * np.sin(xs / 78.0) - 20 * np.sin(xs / 31.0 + 2.0)).astype(int)
    color = np.array([27, 38, 74], np.float32)
    for x in range(W):
        field[ridge[x]:int(H * 0.72), x] = field[ridge[x]:int(H * 0.72), x] * 0.35 + color * 0.65


def paint_village(field: np.ndarray, rng: np.random.Generator, img_draw: ImageDraw.ImageDraw) -> None:
    ground_y = int(H * 0.70)
    field[ground_y:, :] = field[ground_y:, :] * 0.30 + np.array([26, 34, 58], np.float32) * 0.70
    rng2 = np.random.default_rng(21)
    houses = []
    x = 12
    while x < W - 60:
        bw = int(rng2.integers(26, 64))
        bh = int(rng2.integers(20, 46))
        houses.append((x, ground_y + 24, bw, bh))
        x += bw + int(rng2.integers(4, 18))
    for hx, hy, bw, bh in houses:
        img_draw.rectangle([hx, hy - bh, hx + bw, hy + bh // 3], fill=(22, 29, 49))
        roof = rng2.integers(0, 2)
        if roof:
            img_draw.polygon([(hx, hy - bh), (hx + bw, hy - bh), (hx + bw // 2, hy - bh - 12)],
                             fill=(16, 22, 40))
        for _ in range(max(1, bw // 18)):
            wx = hx + int(rng2.integers(4, max(5, bw - 10)))
            wy = hy - int(rng2.integers(6, max(7, bh - 4)))
            img_draw.rectangle([wx, wy, wx + 5, wy + 6], fill=(240, 196, 96))
    # church spire
    spx = int(W * 0.52)
    img_draw.polygon([(spx - 9, ground_y + 30), (spx + 9, ground_y + 30),
                      (spx + 3, ground_y - 74), (spx - 3, ground_y - 74)], fill=(20, 27, 46))
    img_draw.rectangle([spx - 14, ground_y - 6, spx + 14, ground_y + 30], fill=(20, 27, 46))
    img_draw.rectangle([spx - 2, ground_y - 30, spx + 2, ground_y - 18], fill=(240, 196, 96))


def paint_cypress(img_draw: ImageDraw.ImageDraw) -> None:
    def flame(y_top: float, y_bot: float, x_base: float, width_scale: float, color):
        n = 60
        ys = np.linspace(y_top, y_bot, n)
        pts_left, pts_right = [], []
        for i, y in enumerate(ys):
            t = i / (n - 1)
            bulge = np.sin(np.pi * (0.15 + 0.85 * t)) ** 0.8
            sway = 16 * np.sin(t * 5.2 + 0.6)
            half = width_scale * bulge * (1.0 - 0.25 * t)
            xc = x_base + sway
            pts_left.append((xc - half, y))
            pts_right.append((xc + half, y))
        img_draw.polygon(pts_left + list(reversed(pts_right)), fill=color)
        for k in range(14):  # texture streaks
            t = k / 13.0
            y = y_top + (y_bot - y_top) * t
            sw = 16 * np.sin(t * 5.2 + 0.6)
            img_draw.line([(x_base + sw - 3, y), (x_base + sw + 3, y + 9)],
                          fill=(28, 42, 34), width=3)

    flame(H * 0.04, H * 0.78, W * 0.115, 30, (23, 34, 26))
    flame(H * 0.14, H * 0.80, W * 0.175, 20, (20, 30, 24))


def build(seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    field = sky_gradient(rng)
    paint_swirls(field, rng)

    img = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")
    paint_mountains(field)  # before PIL overlay, drawn in numpy field
    img2 = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    draw2 = ImageDraw.Draw(img2, "RGBA")
    paint_village(field, rng, draw2)
    img3 = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))

    # glows need float blending on the current field
    stars = [(60, 96, 9), (150, 60, 11), (232, 120, 8), (300, 208, 13), (368, 84, 10),
             (430, 150, 9), (500, 74, 12), (548, 200, 10), (196, 300, 8), (84, 210, 9),
             (420, 268, 9)]
    for sx, sy, sr in stars:
        paint_glow(field, sx, sy, sr, (255, 232, 120), (214, 178, 96), flare=True)
    paint_glow(field, W * 0.845, H * 0.115, 26, (255, 240, 158), (232, 178, 92), flare=False)
    paint_glow(field, W * 0.845, H * 0.115, 16, (255, 250, 205), (255, 220, 130), flare=False)

    img4 = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    draw4 = ImageDraw.Draw(img4, "RGBA")
    paint_cypress(draw4)

    # gentle canvas grain
    arr = np.asarray(img4, np.float32)
    grain = rng.normal(0, 2.2, size=arr.shape[:2])[:, :, None]
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/input/starry_night_synthetic.png")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    img = build(seed=args.seed)
    img.save(out)
    print(f"synthetic starry night -> {out} ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
