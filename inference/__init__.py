"""Van Gogh Paper-Level Neural Preserving Demo - inference package.

Pipeline stages
---------------
1. ``sam_segment``   : SAM (ViT-H / ViT-B) automatic segmentation -> semantic RGBA layers
2. ``depth_estimate``: ZoeDepth / MiDaS monocular depth -> relative depth map
3. ``optical_flow``  : RAFT dense flow on a depth-synthesised parallax pair
4. ``painterly``     : Hertzmann-style curved brush stroke extraction
5. ``export_web``    : WebGL2 asset bundle + manifest for the live renderer

Every module is importable both as a package member (``inference.x``) and,
for backwards compatibility with the original skeleton, as a flat module
when executed from inside the ``inference/`` directory.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "utils",
    "flow_codec",
    "sam_segment",
    "depth_estimate",
    "optical_flow",
    "painterly",
    "export_web",
    "run_pipeline",
]
