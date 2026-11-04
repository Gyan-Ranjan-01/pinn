"""
Differentiable Dark Channel Prior (DCP) and Airlight Estimation for Outdoor Images.

References:
    He, K., Sun, J., & Tang, X. (2010). Single Image Haze Removal Using Dark Channel Prior.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(12), 2341-2353.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dark_channel_torch(
    image: torch.Tensor,
    patch_size: int = 15,
) -> torch.Tensor:
    """
    Computes the dark channel prior map differentiably using inverted max-pooling:
        J_dark(x) = min_{c in {R,G,B}} ( min_{y in Omega(x)} I^c(y) )

    Args:
        image: (B, 3, H, W) tensor in [0, 1].
        patch_size: Square window kernel size (odd integer).

    Returns:
        dark_channel: (B, 1, H, W) tensor in [0, 1].
    """
    # Step 1: Min across color channels (B, 1, H, W)
    min_channel, _ = torch.min(image, dim=1, keepdim=True)

    # Step 2: Differentiable local minimum via negative max pooling
    pad = patch_size // 2
    dark_channel = -F.max_pool2d(
        -min_channel,
        kernel_size=patch_size,
        stride=1,
        padding=pad,
    )
    return dark_channel


def estimate_atmospheric_light_torch(
    image: torch.Tensor,
    dark_channel: Optional[torch.Tensor] = None,
    top_percent: float = 0.001,
    patch_size: int = 15,
) -> torch.Tensor:
    """
    Estimates the global atmospheric light (airlight A) vector from the brightest pixels
    in the dark channel.

    Args:
        image: (B, 3, H, W) tensor in [0, 1].
        dark_channel: Optional precomputed dark channel. If None, computed with patch_size.
        top_percent: Fraction of brightest dark-channel pixels to consider (e.g., 0.001 = 0.1%).
        patch_size: Patch size if dark_channel is computed.

    Returns:
        airlight: (B, 3, 1, 1) atmospheric light vector in [0, 1].
    """
    B, C, H, W = image.shape
    if dark_channel is None:
        dark_channel = compute_dark_channel_torch(image, patch_size=patch_size)

    num_pixels = H * W
    k = max(1, int(num_pixels * top_percent))

    # Flatten dark channel and image
    dc_flat = dark_channel.view(B, num_pixels)  # (B, N)
    img_flat = image.view(B, C, num_pixels)    # (B, 3, N)

    # Find indices of top-k brightest dark channel pixels
    _, topk_indices = torch.topk(dc_flat, k=k, dim=1)  # (B, k)

    # Gather corresponding RGB values from the image
    topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, C, -1)  # (B, 3, k)
    selected_pixels = torch.gather(img_flat, 2, topk_indices_expanded)   # (B, 3, k)

    # Find the pixel among candidates with highest intensity (L1 norm across channels)
    intensity = selected_pixels.mean(dim=1)  # (B, k)
    best_idx = torch.argmax(intensity, dim=1)  # (B,)

    batch_indices = torch.arange(B, device=image.device)
    airlight_vec = selected_pixels[batch_indices, :, best_idx]  # (B, 3)

    # Clamp to reasonable outdoor sky values [0.5, 1.0]
    airlight_vec = torch.clamp(airlight_vec, 0.5, 1.0)
    return airlight_vec.view(B, 3, 1, 1)


def estimate_transmission_dcp(
    image: torch.Tensor,
    airlight: Optional[torch.Tensor] = None,
    omega: float = 0.95,
    patch_size: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes coarse transmission map via Dark Channel Prior:
        t(x) = 1 - omega * min_{c} ( min_{y in Omega(x)} (I^c(y) / A^c) )

    Args:
        image: (B, 3, H, W) tensor in [0, 1].
        airlight: Optional (B, 3, 1, 1) atmospheric light. If None, estimated.
        omega: Haze retention factor (typically 0.95).
        patch_size: Patch size for min pooling.

    Returns:
        Tuple of:
            transmission: (B, 1, H, W) transmission in [0, 1].
            airlight: (B, 3, 1, 1) atmospheric light used.
    """
    if airlight is None:
        airlight = estimate_atmospheric_light_torch(image, patch_size=patch_size)

    # Normalize image by airlight
    norm_img = image / torch.clamp(airlight, min=1e-3)

    # Compute dark channel of normalized image
    norm_dark = compute_dark_channel_torch(norm_img, patch_size=patch_size)

    # Compute transmission map
    transmission = 1.0 - omega * norm_dark
    return torch.clamp(transmission, 0.05, 1.0), airlight

