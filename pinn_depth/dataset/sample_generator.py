"""
Procedural Outdoor Scene Generator for Testing, Benchmarking, and Demonstration.
Generates realistic outdoor scenes (road, roadside trees/buildings, cars, horizon sky)
with exact ground-truth depth maps.
"""

from typing import Tuple, List
import math
import numpy as np
import cv2
import torch


def generate_synthetic_outdoor_scene(
    width: int = 384,
    height: int = 256,
    num_cars: int = 2,
    num_trees: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a realistic synthetic outdoor driving/street scene and its corresponding
    normalized ground-truth relative depth map in [0, 1].

    Returns:
        image: (H, W, 3) uint8 RGB image
        depth: (H, W) float32 relative depth map in [0.0 (near), 1.0 (far)]
    """
    rng = np.random.default_rng(seed)

    image = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.ones((height, width), dtype=np.float32)

    horizon_y = int(height * 0.45)

    # 1. Sky (from y=0 to horizon_y)
    sky_depth = 1.0
    depth[:horizon_y, :] = sky_depth
    for y in range(horizon_y):
        t = y / max(1, horizon_y)
        # Deep sky blue to warm haze near horizon
        image[y, :, 0] = 0.45 + 0.35 * t  # R
        image[y, :, 1] = 0.65 + 0.25 * t  # G
        image[y, :, 2] = 0.95 - 0.10 * t  # B

    # 2. Road and Ground Plane (from horizon_y to height)
    # Perspective depth: d decreases as y moves closer to the camera (bottom of image)
    for y in range(horizon_y, height):
        rel_y = (y - horizon_y) / (height - horizon_y)  # 0 at horizon, 1 at camera bottom
        # Perspective depth formula: d ~ 1 / y_screen
        # Scaled to [0.05, 0.95]
        d_val = float(0.05 + 0.90 * math.exp(-2.5 * rel_y))
        depth[y, :] = d_val

        # Asphalt road color with subtle texture gradient
        base_gray = 0.22 + 0.08 * (1.0 - rel_y)
        image[y, :, :] = base_gray

    # Road lane markings
    lane_center = width // 2
    for y in range(horizon_y, height, 16):
        rel_y = (y - horizon_y) / (height - horizon_y)
        w_lane = max(2, int(6 * rel_y))
        y_end = min(height, y + 8)
        image[y:y_end, lane_center - w_lane : lane_center + w_lane, :] = 0.85

    # 3. Mountains / Distant Buildings near horizon
    for _ in range(5):
        bx = rng.integers(10, width - 60)
        bw = rng.integers(30, 80)
        bh = rng.integers(20, 50)
        by = horizon_y - bh
        b_depth = float(rng.uniform(0.85, 0.95))

        # Building color
        b_color = rng.uniform(0.35, 0.65, size=3)
        image[by:horizon_y, bx : bx + bw, :] = b_color
        depth[by:horizon_y, bx : bx + bw] = b_depth

    # 4. Trees along roadside
    for i in range(num_trees):
        side = -1 if i % 2 == 0 else 1
        t_y = horizon_y + int((height - horizon_y) * (0.2 + 0.6 * (i / num_trees)))
        rel_y = (t_y - horizon_y) / (height - horizon_y)
        t_depth = float(0.05 + 0.90 * math.exp(-2.5 * rel_y))

        tree_h = int(60 * rel_y)
        tree_w = int(24 * rel_y)
        if side < 0:
            tx = int(width * 0.15 - tree_w / 2)
        else:
            tx = int(width * 0.85 - tree_w / 2)

        # Trunk
        trunk_w = max(2, int(tree_w * 0.2))
        trunk_h = int(tree_h * 0.4)
        ty_base = t_y
        ty_top = t_y - trunk_h
        image[ty_top:ty_base, tx + tree_w // 2 - trunk_w // 2 : tx + tree_w // 2 + trunk_w // 2, :] = [0.25, 0.15, 0.08]

        # Foliage (Green cluster)
        center_f = (tx + tree_w // 2, ty_top)
        radius = max(6, int(tree_w * 0.7))
        cv2.circle(image, center_f, radius, (0.15, 0.45, 0.15), -1)
        cv2.circle(depth, center_f, radius, t_depth, -1)

    # 5. Foreground / Midground Cars
    for j in range(num_cars):
        car_y = horizon_y + int((height - horizon_y) * (0.45 + 0.4 * (j / max(1, num_cars))))
        rel_y = (car_y - horizon_y) / (height - horizon_y)
        c_depth = float(0.05 + 0.90 * math.exp(-2.5 * rel_y))

        car_w = int(80 * rel_y)
        car_h = int(35 * rel_y)
        car_x = int(width * (0.35 + 0.2 * (j % 2)) - car_w // 2)

        car_color = [0.8, 0.1, 0.1] if j % 2 == 0 else [0.1, 0.3, 0.8]

        # Car body
        image[car_y - car_h : car_y, car_x : car_x + car_w, :] = car_color
        depth[car_y - car_h : car_y, car_x : car_x + car_w] = c_depth

        # Cabin / Roof
        roof_w = int(car_w * 0.6)
        roof_h = int(car_h * 0.5)
        roof_x = car_x + (car_w - roof_w) // 2
        roof_y = car_y - car_h - roof_h
        image[roof_y : car_y - car_h, roof_x : roof_x + roof_w, :] = [0.2, 0.2, 0.2]
        depth[roof_y : car_y - car_h, roof_x : roof_x + roof_w] = c_depth

    # Convert to uint8 RGB image
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    depth_float = np.clip(depth, 0.0, 1.0).astype(np.float32)

    return img_uint8, depth_float


def generate_benchmark_dataset(
    output_dir: str,
    num_samples: int = 24,
) -> List[Tuple[str, str]]:
    """
    Generates a folder of synthetic benchmark outdoor images and ground-truth depth maps.
    """
    import os
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    file_pairs = []
    for i in range(num_samples):
        img, dep = generate_synthetic_outdoor_scene(width=384, height=256, seed=100 + i)

        img_path = os.path.join(output_dir, "images", f"scene_{i:03d}.jpg")
        dep_path = os.path.join(output_dir, "depth", f"scene_{i:03d}.npy")

        # Save image (convert RGB to BGR for cv2)
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(dep_path, dep)

        file_pairs.append((img_path, dep_path))

    return file_pairs

