"""
Test Edge-Guided High-Definition Physics Depth Filter.
Applies Guided Image Filtering to transfer photorealistic edges directly
into the relative depth map, matching the clarity in Das et al. (ICVGIP 2012).
"""

import cv2
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


def get_das_paper_colormap():
    colors = [
        (1.0, 1.0, 0.0),    # 0.0: Bright Yellow (Nearest, foreground)
        (1.0, 0.5, 0.0),    # 0.25: Orange
        (0.85, 0.1, 0.3),   # 0.50: Crimson / Magenta
        (0.2, 0.3, 0.85),   # 0.75: Royal Blue
        (0.05, 0.05, 0.2),  # 0.95: Deep Indigo
        (0.0, 0.0, 0.05),   # 1.0: Dark Black (Farthest, sky)
    ]
    return LinearSegmentedColormap.from_list("das_icvgip", colors, N=256)


def fast_guided_filter(guide_gray: np.ndarray, depth_src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    Fast, edge-preserving Guided Filter (He et al., IEEE TPAMI).
    Transfers high-frequency guidance image edges into the depth map.
    """
    I = guide_gray.astype(np.float32)
    P = depth_src.astype(np.float32)

    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.blur(I, ksize)
    mean_P = cv2.blur(P, ksize)
    mean_IP = cv2.blur(I * P, ksize)
    cov_IP = mean_IP - mean_I * mean_P

    mean_II = cv2.blur(I * I, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I

    mean_a = cv2.blur(a, ksize)
    mean_b = cv2.blur(b, ksize)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


def generate_high_def_depth(image_path: str = "samples/garden.jpg"):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 1. High capacity depth feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval().to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    raw_disp = pred.cpu().numpy()
    d_min, d_max = float(raw_disp.min()), float(raw_disp.max())
    norm_disp = (raw_disp - d_min) / (d_max - d_min + 1e-7)

    # Invert so 0.0 is closest (foreground) and 1.0 is farthest (sky)
    coarse_depth = np.clip(1.0 - norm_disp, 0.0, 1.0)

    # 2. Edge-Guided High-Frequency Filter
    guide_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sharp_depth = fast_guided_filter(guide_gray, coarse_depth, radius=6, eps=1e-3)

    # 3. Apply the exact Das et al. (ICVGIP 2012) Colormap
    cmap_das = get_das_paper_colormap()

    # In the paper:
    # Nearest = Yellow (value = 0.0)
    # Farthest = Black / Deep Blue (value = 1.0)
    vis_rgb = cmap_das(sharp_depth)[:, :, :3]
    vis_bgr = (cv2.cvtColor((vis_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Also generate Turbo colormap for comparison
    vis_turbo_bgr = cv2.applyColorMap(((1.0 - sharp_depth) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)

    cv2.imwrite("samples/garden_das_colormapped.png", vis_bgr)
    cv2.imwrite("samples/garden_sharp_turbo.png", vis_turbo_bgr)
    print("Generated samples/garden_das_colormapped.png and samples/garden_sharp_turbo.png")


if __name__ == "__main__":
    generate_high_def_depth()
