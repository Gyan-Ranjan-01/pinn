"""
API utilities for image encoding, colormaps, and 3D heightfield generation.
"""

from typing import Tuple, Dict, Any, List
import io
import base64
import numpy as np
import cv2
from PIL import Image


def numpy_to_base64_png(array_rgb: np.ndarray) -> str:
    """
    Converts (H, W, 3) uint8 RGB array to a base64 data URL.
    """
    pil_img = Image.fromarray(array_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def depth_to_point_cloud_json(
    depth: np.ndarray,
    image_rgb: np.ndarray,
    max_resolution: int = 120,
) -> Dict[str, Any]:
    """
    Downsamples depth map and generates 3D point cloud / vertex grid
    suitable for Three.js WebGL visualization.
    """
    H, W = depth.shape
    # Scale down for lightweight web streaming
    scale = min(1.0, max_resolution / max(H, W))
    new_w = max(16, int(W * scale))
    new_h = max(16, int(H * scale))

    d_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Normalize depth so closer objects have positive height / displacement
    # 0 is near, 1 is far -> height = (1 - depth)
    height_grid = (1.0 - d_small) * 1.5

    positions: List[float] = []
    colors: List[float] = []

    for y in range(new_h):
        for x in range(new_w):
            # Normalized 3D space: X in [-1, 1], Y in [-aspect, aspect], Z is depth
            norm_x = (x / (new_w - 1) - 0.5) * 2.0
            norm_y = -(y / (new_h - 1) - 0.5) * (2.0 * new_h / new_w)
            norm_z = float(height_grid[y, x])

            r, g, b = img_small[y, x] / 255.0

            positions.extend([float(norm_x), float(norm_y), float(norm_z)])
            colors.extend([float(r), float(g), float(b)])

    return {
        "width": new_w,
        "height": new_h,
        "num_points": new_w * new_h,
        "positions": positions,
        "colors": colors,
    }

