# Van Gogh Neural Original Preserving

A static web demo of *The Starry Night* driven by an offline neural CV pipeline.
All three neural assets (MobileSAM layers / MiDaS depth / RAFT optical flow) are
generated offline at build time. The site itself is pure static, zero-build, and
ready to deploy on Netlify as-is.

## What You See

- **Depth parallax**: moving the pointer displaces scene regions according to the
  MiDaS depth map (strongest in the village band, weakest for celestial bodies)
- **Flow swirls**: the RAFT optical-flow field drives a reciprocating swirl of the
  night sky, with a dual-frequency temporal pulse
- **Layered motion**: the six SAM segmentation layers (sky / moon / stars / cypress /
  village / mountains) control per-region motion amplitude, with a subtle twinkle
  added to the star layer
- **Aspect-correct letterbox**: the painting is never stretched or cropped — the full
  artwork stays visible with black bars filling the remaining viewport
- **Graceful fallback**: if any asset fails to load, main.js switches to a procedural
  swirl that is still clearly animated and aspect-correct — the page never degrades
  to a static, distorted image
- **Status badge**: a small self-fading badge at the bottom-left shows whether the
  neural pipeline is active (green) or missing (orange), which makes deployment
  problems obvious at a glance

## Repository Layout

```
index.html                  # entry point (three.js CDN + main.js)
main.js                     # neural-asset consumer shader (with fallback)
starry-night.jpg            # source image, 1280x1014
netlify.toml                # static deploy config (publish = ".")
cv_pipeline/
  scripts/                  # offline generation scripts (documentation only,
    gen_depth.py            #   paths are machine-local)
    gen_flow.py            #   RAFT-small optical flow (swirl frame-pair method)
    gen_layers.py          #   MobileSAM point-prompted layer masks
  assets/
    manifest.json           # master asset manifest (models / encodings / stats)
    depth/                  # depth maps (8-bit / 16-bit PNG + stats)
    flow/                   # optical flow (.flo + RGB-encoded PNG + viz + stats)
    layers/                 # six segmentation masks + metadata + preview
```

## cv_pipeline/assets Encoding Conventions

- **depth_midas_small.png**: 8-bit grayscale, 255 = nearest (village band),
  0 = farthest (moon / stars). Raw MiDaS output, min-max normalized and preserved
- **flow_raft_small.flo**: standard Middlebury format (float32, u then v per pixel),
  full 1280x1014 resolution
- **flow_raft_rgb.png**: GPU-texture-ready encoding, `flow_px = (texel.rg - 0.5) * 32.0`
  (128 = zero flow, B channel = |f| / 16)
- **layer_0..5_*.png**: mutually exclusive binary masks (255 = pixel belongs to the
  layer); the six layers partition the image exactly

## Generation Environment

- Python 3.12 + torch 2.14 (CPU) + torchvision + timm + opencv
- Model weights: MiDaS v2.1 small (82 MB) / MobileSAM vit_t (40 MB) /
  RAFT-small C+T weights (hosted officially by torchvision)

## Deploying to Netlify

1. Push the CONTENTS of this folder to GitHub — `index.html` must sit at the
   repository root, together with `main.js`, `starry-night.jpg`, `netlify.toml`
   and the whole `cv_pipeline/` folder
2. Import the repository on Netlify — no build command needed; the publish
   directory is the repo root (already configured via netlify.toml)
3. Alternatively, drag-and-drop this folder at https://app.netlify.com/drop

### Deployment checklist (verify on GitHub after uploading)

- [ ] `index.html` is at the repository ROOT (not inside a subfolder)
- [ ] The `cv_pipeline/assets/` tree exists with these files:
      `manifest.json`, `depth/depth_midas_small.png`, `flow/flow_raft_rgb.png`,
      `layers/layer_0_sky.png` .. `layer_5_mountains.png`
- [ ] No `.zip` file was committed — GitHub stores zips as a single binary blob,
      they are never extracted automatically
- [ ] The production branch in Netlify matches the branch you pushed to

## Troubleshooting

- **Orange badge "NEURAL PIPELINE: NOT LOADED"** — the `cv_pipeline/` folder did
  not make it into the repository (the most common cause is uploading the files
  as a zip, or skipping the nested folders in the GitHub web uploader). Re-upload
  the `cv_pipeline/` folder, commit, and Netlify will redeploy automatically.
  The page keeps working in the meantime via the procedural fallback.
- **Stretched / cropped image** — fixed by design: the demo now letterboxes the
  painting so it is always fully visible and undistorted on any screen.
- **No animation visible** — both modes animate continuously; if the page is
  truly static, open the browser console — `window.__vanGoghStatus` reports the
  active mode and any missing files.
- **404 on the site root** — `index.html` is not at the repository root (see the
  deployment checklist above).
