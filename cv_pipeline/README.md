# cv_pipeline — PyTorch asset generation

Everything the web demo displays is precomputed here by real models — no
hand-drawn animation frames ship with the site. All pixel-level math runs in
PyTorch tensors; OpenCV/PIL are used only for file I/O and blob detection.

## Scripts (run in this order)

| # | Script | Model | Output |
|---|--------|-------|--------|
| 1 | `gen_depth.py` | MiDaS v2.1 small (efficientnet_lite3) | `assets/depth/depth_midas_small.png` |
| 2 | `gen_layers.py` | MobileSAM vit_t (TinyViT encoder) | `assets/layers/layer_0..5_*.png` |
| 3 | `gen_flow.py` | RAFT-small (torchvision, C+T v2 weights) | `assets/flow/flow_raft_phases.png` |

`gen_flow.py` reads the layer masks from step 2 (region-weighted motion model),
so the order matters. Total runtime on a plain CPU laptop: roughly 1–2 minutes.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
```

`requirements.txt` covers torch, torchvision, timm and OpenCV. MobileSAM is not
on PyPI, so install it separately:

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
# checkpoint (~40 MB) for gen_layers.py:
curl -L -o mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

`gen_depth.py` and `gen_flow.py` download their own weights through the
official entry points (`torch.hub` for MiDaS ≈ 82 MB, torchvision for RAFT
≈ 1 MB) on first run.

## Regenerating all assets

```bash
python cv_pipeline/gen_depth.py
python cv_pipeline/gen_layers.py
python cv_pipeline/gen_flow.py
```

Each script resolves paths relative to the repository root, so you can run
them from anywhere inside the repo. After they finish, open `index.html`
(or redeploy) — the player picks the new assets up automatically.

## Sanity numbers (this commit)

- Depth: village band is the nearest plane, moon sky the farthest (relative
  inverse depth, MiDaS convention).
- Layers: sky ≈ 65%, cypress ≈ 12%, mountains ≈ 12%, village ≈ 7%,
  stars ≈ 2%, moon ≈ 1% of the canvas.
- Flow cycle: 16 phases; peak phases move 98–99% of all pixels by more than
  1 px (mean magnitude ≈ 9.5 px); phase 0 is bit-identical to the original
  painting, which makes the loop seamless. Per-region mean displacement at
  the reference phase: sky 8.0 / stars 9.8 / mountains 3.9 / village 2.2 /
  cypress 1.6 / moon 1.3 px.
