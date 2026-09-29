# Van Gogh Neural Original Preserving

A static web demo of *The Starry Night* driven by an offline neural CV pipeline.
All three neural assets (MobileSAM layers / MiDaS depth / RAFT optical flow) are
generated offline at build time. The site itself is pure static, zero-build, and
ready to deploy on Netlify as-is.

## What You See

The **entire painting moves** — every region has its own animated motion:

- **Sky**: RAFT-flow-driven swirls (the famous double vortex + the great wave
  current) flowing back and forth, strongest in the sky, with the moon glow
  gently breathing and the stars twinkling
- **Cypress**: sways with a pivot at its base, the crown moving the most
- **Village**: slow lateral drift with warm window-light flicker
- **Mountains**: gentle lateral drift, out of phase with the village
- **Depth parallax**: moving the pointer displaces scene regions according to the
  MiDaS depth map (strongest in the village band)
- **No lockstep**: each layer pulses with its own temporal phase, and a
  procedural wave is blended over the neural flow so the whole canvas keeps
  breathing everywhere
- **Aspect-correct letterbox**: the painting is never stretched or cropped — the
  full artwork stays visible with black bars filling the remaining viewport
- **Graceful fallback**: if any asset fails to load, main.js switches to a
  procedural swirl that is still clearly animated and aspect-correct
- **Status badge**: a small self-fading badge at the bottom-left shows whether the
  neural pipeline is active (green) or missing (orange)

## Repository Layout

```
index.html                  # entry point (three.js CDN + main.js)
main.js                     # neural-asset consumer shader (with fallback)
starry-night.jpg            # source image, 1280x1014
netlify.toml                # static deploy config (publish = ".")
cv_pipeline/
  scripts/                  # offline generation scripts (documentation only,
    gen_depth.py            #   paths are machine-local)
    gen_flow.py            #   RAFT-small optical flow (region-weighted field)
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
  (128 = zero flow, B channel = |f| / 16). The v4 field moves 97.8% of all pixels
  (mean 5.5 px): sky ~7 px, cypress / village / mountains ~3 px each
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
- **Only one region moving** — fixed in v4: the flow field now covers the whole
  canvas (97.8% of pixels), the per-layer amplitude suppression was removed, and
  a procedural wave is blended over the neural flow. If you still see a single
  moving spot, your deploy is running an old `main.js` or an old
  `flow_raft_rgb.png` — re-upload both.
- **Stretched / cropped image** — fixed by design: the demo letterboxes the
  painting so it is always fully visible and undistorted on any screen.
- **404 on the site root** — `index.html` is not at the repository root (see the
  deployment checklist above).
