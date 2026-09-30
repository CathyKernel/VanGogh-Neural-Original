# Model checkpoints

SAM checkpoints are stored here and are **not** committed / packaged -
fetch them with:

    python scripts/download_models.py --profile cpu     # SAM ViT-B ( 375 MB )
    PROFILE=paper ./scripts/download_models.sh          # SAM ViT-H ( 2.4 GB )

| file                     | size    | profile | source                                                        |
|--------------------------|---------|---------|---------------------------------------------------------------|
| `sam_vit_h_4b8939.pth`   | 2.4 GB  | paper   | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| `sam_vit_l_0b3195.pth`   | 1.2 GB  | -       | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| `sam_vit_b_01ec64.pth`   | 375 MB  | cpu     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

Depth ( ZoeD_NK / MiDaS_small ) and flow ( RAFT-large ) weights are fetched
automatically by `torch.hub` / `torchvision` into the torch cache on first
inference - nothing to place here manually.

If no SAM checkpoint can be obtained, the pipeline degrades gracefully to a
classical Lab-k-means + connected-components segmenter ( logged clearly ).
