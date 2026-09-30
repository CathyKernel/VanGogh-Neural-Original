#!/usr/bin/env bash
# Download model checkpoints for the Van Gogh neural-preserving pipeline.
#
#   ./scripts/download_models.sh                 # auto profile ( CPU-friendly )
#   PROFILE=paper ./scripts/download_models.sh   # paper-level ViT-H ( 2.4 GB )
#   WARMUP=1 ./scripts/download_models.sh        # also pre-fetch depth + RAFT weights
#
# Only the SAM checkpoints need explicit management. ZoeDepth / MiDaS and
# RAFT weights download automatically through torch.hub / torchvision on
# first inference ( or right now when WARMUP=1 ).
set -euo pipefail

PROFILE="${PROFILE:-auto}"
WARMUP="${WARMUP:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA=()
if [[ "${WARMUP}" == "1" ]]; then
  EXTRA+=(--warmup)
fi

python3 "${SCRIPT_DIR}/download_models.py" --profile "${PROFILE}" "${EXTRA[@]+"${EXTRA[@]}"}"

cat <<'EOF'

Manual URLs if you prefer to fetch weights yourself:

  SAM ViT-H : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  (2.4 GB)
  SAM ViT-L : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  (1.2 GB)
  SAM ViT-B : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  (375 MB)
  ZoeDepth  : https://github.com/isl-org/ZoeDepth   ( torch.hub: ZoeD_NK )
  MiDaS     : https://github.com/isl-org/MiDaS      ( torch.hub: MiDaS_small )
  RAFT      : torchvision.models.optical_flow       ( raft_large weights )

Place SAM .pth files into ./checkpoints/ then rerun the pipeline.
EOF
