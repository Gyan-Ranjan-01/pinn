"""
Physics of Outdoor Atmospheric Scattering based on:
    Das, J. K., Biswas, A., & Saha, S. K. (2012).
    "Depth From Images of External Outdoor Scenes"
    Proceedings of the Eighth Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP '12), ACM.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
import torch


def compute_luminance_das(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes luminance according to Das et al. (ICVGIP 2012) Section 3:
        C(x, y) = 0.33 * R + 0.56 * G + 0.11 * B
    """
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    C = 0.33 * R + 0.56 * G + 0.11 * B
    return np.clip(C, 1.0, 255.0)


def compute_das_attenuation_beta(
    luminance: np.ndarray,
    c_gm: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Computes empirical attenuation coefficient beta using Das et al. Equations (8) & (19):
        C_tilde(x, y) = C(x, y) * (2 - C/C_gm) - C_gm * (1 - C/C_gm)
        beta(x, y) = - sqrt(C / C_gm) * ln(C / C_tilde)

    Returns:
        beta_scene: Scalar constant beta for the scene (typically 0.33 - 0.56)
        beta_map: Per-pixel beta distribution
    """
    if c_gm is None:
        c_gm = float(np.percentile(luminance, 99.5))
        c_gm = max(c_gm, 180.0)

    ratio = np.clip(luminance / c_gm, 1e-4, 1.0)
    # Eqn (19): unscattered luminance
    c_tilde = luminance * (2.0 - ratio) - c_gm * (1.0 - ratio)
    c_tilde = np.clip(c_tilde, 1.0, 255.0)

    # Eqn (8): per-pixel attenuation beta
    ratio_c = np.clip(luminance / c_tilde, 1e-4, 1.0)
    beta_map = -np.sqrt(ratio) * np.log(ratio_c + 1e-6)

    # Positive betas represent physical attenuation
    pos_beta = beta_map[beta_map > 0]
    if len(pos_beta) > 100:
        beta_scene = float(np.median(pos_beta))
    else:
        beta_scene = 0.45

    beta_scene = float(np.clip(beta_scene, 0.25, 0.65))
    return beta_scene, beta_map


def fast_guided_filter(
    guide_gray: np.ndarray,
    depth_src: np.ndarray,
    radius: int = 6,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    High-frequency edge-preserving Guided Filter (He et al., IEEE TPAMI).
    Transfers photorealistic boundaries (leaves, flower petals, architecture edges)
    from the RGB guide image directly into the depth map, eliminating blur.
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


def get_das_paper_colormap() -> LinearSegmentedColormap:
    """
    Exact colormap from Das et al. (ICVGIP 2012) Figures 3, 7, 9, 10, 11:
        - 0.0 (Nearest to viewer): Bright Yellow (#FFFF00)
        - 0.20: Warm Gold / Orange (#FFA500)
        - 0.45: Crimson Red / Magenta (#E11D48)
        - 0.70: Royal Blue (#2563EB)
        - 0.85: Deep Indigo (#1E1B4B)
        - 1.00 (Farthest / Sky): Dark Black (#000000)
    """
    colors = [
        (1.0, 1.0, 0.0),    # 0.00: Bright Yellow (Nearest, foreground)
        (1.0, 0.55, 0.0),   # 0.20: Orange / Gold
        (0.85, 0.1, 0.25),  # 0.45: Crimson / Magenta
        (0.15, 0.35, 0.9),  # 0.70: Royal Blue
        (0.06, 0.08, 0.3),  # 0.85: Deep Indigo
        (0.0, 0.0, 0.02),   # 1.00: Dark Black (Farthest, sky)
    ]
    return LinearSegmentedColormap.from_list("das_icvgip", colors, N=256)


# Precompute 256-color lookup table for fast RGB mapping
_DAS_CMAP = get_das_paper_colormap()
DAS_LUT_RGB = (_DAS_CMAP(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)


def get_distinct_colormap() -> LinearSegmentedColormap:
    """
    High-Distinction Multi-Hue Colormap for fine-grained object delineation:
    Transitions across distinct perceptual color bands so adjacent items
    at slightly different depths are immediately recognizable.
    """
    colors = [
        (1.0, 1.0, 0.0),    # 0.00: Radiant Yellow (Immediate foreground)
        (1.0, 0.60, 0.0),   # 0.15: Warm Amber
        (0.95, 0.18, 0.12), # 0.30: Coral Red
        (0.85, 0.10, 0.55), # 0.45: Magenta / Crimson
        (0.45, 0.15, 0.85), # 0.60: Violet
        (0.12, 0.50, 0.95), # 0.75: Azure Blue
        (0.05, 0.18, 0.55), # 0.90: Deep Navy
        (0.00, 0.00, 0.02), # 1.00: Dark Black (Sky / Horizon)
    ]
    return LinearSegmentedColormap.from_list("distinct", colors, N=256)


_DISTINCT_CMAP = get_distinct_colormap()
DISTINCT_LUT_RGB = (_DISTINCT_CMAP(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)


def colorize_das_paper(depth_norm: np.ndarray) -> np.ndarray:
    """
    Colorizes normalized depth map [0, 1] using the exact Das et al. (ICVGIP 2012) colormap:
    Nearest (0.0) is radiant yellow; Farthest (1.0) is black.
    """
    d_idx = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return DAS_LUT_RGB[d_idx]


def colorize_distinct(depth_norm: np.ndarray) -> np.ndarray:
    """
    Colorizes normalized depth map [0, 1] using the Multi-Hue High-Distinction colormap.
    """
    d_idx = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return DISTINCT_LUT_RGB[d_idx]


def get_flir_ironbow_colormap() -> LinearSegmentedColormap:
    """
    Iconic FLIR Thermal Ironbow Colormap:
    - 0.00: White-Hot (#FFFFFF) (Closest, hottest thermal signature)
    - 0.08: Pale Radiant Yellow (#FEFFA5)
    - 0.20: Golden Amber (#F59E0B)
    - 0.38: Intense Flame Orange (#FB923C)
    - 0.55: Crimson Magenta (#E11D48)
    - 0.72: Deep Purple / Violet (#7C3AED)
    - 0.88: Midnight Blue (#1E1B4B)
    - 1.00: Deep Black (#020617) (Farthest / Coldest / Sky)
    """
    colors = [
        (1.0, 1.0, 1.0),    # 0.00: White-Hot
        (1.0, 0.95, 0.40),  # 0.08: Pale Yellow
        (1.0, 0.70, 0.00),  # 0.20: Golden Amber
        (0.95, 0.28, 0.05), # 0.38: Flame Orange
        (0.75, 0.05, 0.35), # 0.55: Crimson Magenta
        (0.35, 0.05, 0.55), # 0.72: Deep Purple
        (0.08, 0.12, 0.40), # 0.88: Midnight Blue
        (0.01, 0.01, 0.03), # 1.00: Deep Black
    ]
    return LinearSegmentedColormap.from_list("flir_ironbow", colors, N=256)


def get_flir_rainbow_colormap() -> LinearSegmentedColormap:
    """
    FLIR Rainbow High-Contrast Thermal Colormap:
    White -> Crimson Red -> Flame Orange -> Yellow -> Neon Green -> Cyan -> Royal Blue -> Pitch Black
    """
    colors = [
        (1.0, 1.0, 1.0),    # 0.00: White-Hot
        (1.0, 0.15, 0.15),  # 0.15: Crimson Red
        (1.0, 0.60, 0.00),  # 0.30: Flame Orange
        (1.0, 0.95, 0.00),  # 0.45: Yellow
        (0.1, 0.85, 0.20),  # 0.60: Neon Green
        (0.0, 0.75, 1.00),  # 0.75: Cyan
        (0.1, 0.15, 0.85),  # 0.90: Royal Blue
        (0.0, 0.00, 0.02),  # 1.00: Pitch Black
    ]
    return LinearSegmentedColormap.from_list("flir_rainbow", colors, N=256)


_IRONBOW_CMAP = get_flir_ironbow_colormap()
IRONBOW_LUT_RGB = (_IRONBOW_CMAP(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)

_RAINBOW_CMAP = get_flir_rainbow_colormap()
RAINBOW_LUT_RGB = (_RAINBOW_CMAP(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)


def colorize_flir_ironbow(depth_norm: np.ndarray) -> np.ndarray:
    """Colorizes normalized depth map [0, 1] using FLIR Ironbow thermal palette."""
    d_idx = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return IRONBOW_LUT_RGB[d_idx]


def colorize_flir_rainbow(depth_norm: np.ndarray) -> np.ndarray:
    """Colorizes normalized depth map [0, 1] using FLIR Rainbow HC thermal palette."""
    d_idx = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return RAINBOW_LUT_RGB[d_idx]


def apply_flir_msx_detail_embossing(
    thermal_rgb: np.ndarray,
    img_rgb: np.ndarray,
    strength: float = 0.45,
) -> np.ndarray:
    """
    Applies FLIR MSX (Multi-Spectral Dynamic Imaging) detail embossing:
    Extracts high-frequency visual edge relief from the RGB luminance and
    embosses it directly onto the thermal depth map. This preserves razor-sharp
    physical contours (petals, veins, stems, textural edges) with authentic
    thermal camera clarity.
    """
    if strength <= 0.001:
        return thermal_rgb

    # Extract luminance from RGB image
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # High-pass filter via unsharp masking
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.3)
    highpass = gray - blur

    # Edge relief multiplier
    edge_relief = np.clip(0.5 + 2.4 * highpass, 0.0, 1.0)

    # Modulate luminance while preserving thermal color saturation
    thermal_float = thermal_rgb.astype(np.float32) / 255.0
    modulator = (1.0 - strength) + (2.0 * strength) * edge_relief[:, :, np.newaxis]
    fused = np.clip(thermal_float * modulator, 0.0, 1.0)
    return (fused * 255.0).astype(np.uint8)



def enhance_depth_contrast(
    depth: np.ndarray,
    gamma: float = 1.35,
    use_clahe: bool = False,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Applies adaptive percentile stretching and non-linear tone expansion to prevent
    foreground object 'bunching' when distant background/sky is present.
    """
    # Robust percentile bounds
    val_low = np.percentile(depth, p_low)
    val_high = np.percentile(depth, p_high)
    if val_high <= val_low:
        val_high = val_low + 1e-6

    # Normalization across active scene content
    stretched = np.clip((depth - val_low) / (val_high - val_low), 0.0, 1.0)

    # Gamma expansion: spreads foreground depth values across a wider dynamic range
    if abs(gamma - 1.0) > 0.01:
        stretched = np.power(stretched, gamma)

    # Optional local adaptive contrast enhancement (CLAHE)
    if use_clahe:
        u8 = (stretched * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        stretched = clahe.apply(u8).astype(np.float32) / 255.0

    return np.clip(stretched, 0.0, 1.0)


def overlay_depth_boundaries(
    vis_rgb: np.ndarray,
    depth: np.ndarray,
    threshold: float = 0.045,
    outline_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Computes depth-discontinuity gradients and overlays crisp 1px boundaries
    around individual scene objects so each item is clearly outlined and recognizable.
    """
    # Depth gradient in x and y
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Threshold for sharp depth step (1px crisp edge without heavy dilation)
    edges = (mag > threshold).astype(np.uint8)

    out = vis_rgb.copy()
    out[edges > 0] = outline_color
    return out


