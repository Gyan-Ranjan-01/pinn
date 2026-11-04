"""
Unit tests for the physics and loss modules.
"""

import pytest
import torch
from pinn_depth.physics import (
    depth_to_transmission,
    transmission_to_depth,
    synthesize_hazy_image,
    recover_radiance,
    AtmosphericScatteringLayer,
    compute_dark_channel_torch,
    estimate_atmospheric_light_torch,
    estimate_transmission_dcp,
    RelativeDepthRankingLoss,
    EdgeAwareSmoothnessLoss,
    ScaleAndShiftInvariantLoss,
    PhysicsConsistencyLoss,
    PINNCombinedLoss,
)


def test_depth_to_transmission_and_inversion():
    depth = torch.tensor([[[[0.0, 0.5], [1.0, 0.2]]]], dtype=torch.float32)
    beta = 1.5

    trans = depth_to_transmission(depth, beta=beta)
    assert trans.shape == depth.shape
    assert (trans > 0.0).all() and (trans <= 1.0).all()

    # Closest object (d=0) should have maximum transmission (t=1)
    assert torch.isclose(trans[0, 0, 0, 0], torch.tensor(1.0), atol=1e-3)
    # Farthest object (d=1.0) should have lower transmission
    assert trans[0, 0, 1, 0] < trans[0, 0, 0, 1]

    # Invert back to depth
    rec_depth = transmission_to_depth(trans, beta=beta)
    assert torch.allclose(depth, rec_depth, atol=1e-3)


def test_atmospheric_scattering_synthesis_and_recovery():
    B, C, H, W = 2, 3, 32, 32
    radiance = torch.rand(B, C, H, W)
    depth = torch.rand(B, 1, H, W)
    airlight = torch.tensor([0.9, 0.95, 0.9]).view(1, 3, 1, 1).expand(B, -1, -1, -1)

    hazy, trans = synthesize_hazy_image(radiance, depth, airlight=airlight, beta=1.0)
    assert hazy.shape == (B, C, H, W)
    assert trans.shape == (B, 1, H, W)
    assert (hazy >= 0.0).all() and (hazy <= 1.0).all()

    # Radiance recovery
    rec_rad = recover_radiance(hazy, trans, airlight=airlight, t_min=0.01)
    assert rec_rad.shape == (B, C, H, W)
    assert torch.allclose(radiance, rec_rad, atol=1e-2)


def test_dark_channel_and_airlight():
    B, C, H, W = 2, 3, 48, 48
    image = torch.rand(B, C, H, W)

    dc = compute_dark_channel_torch(image, patch_size=7)
    assert dc.shape == (B, 1, H, W)
    assert (dc >= 0.0).all() and (dc <= 1.0).all()

    airlight = estimate_atmospheric_light_torch(image, dark_channel=dc)
    assert airlight.shape == (B, 3, 1, 1)
    assert (airlight >= 0.5).all() and (airlight <= 1.0).all()

    t_dcp, _ = estimate_transmission_dcp(image, airlight=airlight, patch_size=7)
    assert t_dcp.shape == (B, 1, H, W)


def test_loss_gradients_and_differentiability():
    B, C, H, W = 2, 3, 32, 32
    pred_depth = torch.rand(B, 1, H, W, requires_grad=True)
    gt_depth = torch.rand(B, 1, H, W)
    image = torch.rand(B, C, H, W)

    combined_loss = PINNCombinedLoss(is_physics_informed=True)
    losses = combined_loss(pred_depth, image, gt_depth=gt_depth)

    assert "total" in losses
    assert "rank" in losses
    assert "smooth" in losses
    assert "ssi" in losses
    assert "phys" in losses

    # Test backward gradient flow
    losses["total"].backward()
    assert pred_depth.grad is not None
    assert torch.isfinite(pred_depth.grad).all()
    assert pred_depth.grad.abs().sum() > 0


def test_depth_enhancement_and_distinct_colorization():
    import numpy as np
    from pinn_depth.physics.das_physics import (
        enhance_depth_contrast,
        colorize_distinct,
        colorize_das_paper,
        overlay_depth_boundaries,
    )

    # Synthetic clustered depth map (simulating foreground flowers with sky)
    depth = np.full((100, 100), 0.45, dtype=np.float32)
    depth[20:50, 20:50] = 0.40   # Foreground flower
    depth[0:20, :] = 0.99        # Sky
    depth[80:100, 40:60] = 0.35  # Closest pansy

    enhanced = enhance_depth_contrast(depth, gamma=1.35, use_clahe=False)
    assert enhanced.shape == (100, 100)
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    # Closer item should have lower depth value
    assert enhanced[90, 50] < enhanced[30, 30] < enhanced[10, 10]

    # Test Distinct Colormap
    colored_rgb = colorize_distinct(enhanced)
    assert colored_rgb.shape == (100, 100, 3)
    assert colored_rgb.dtype == np.uint8

    # Test Depth Boundary Overlay
    outlined = overlay_depth_boundaries(colored_rgb, enhanced, threshold=0.03)
    assert outlined.shape == (100, 100, 3)
    # White boundary pixels should be present along the edge of the foreground flower
    assert (outlined == 255).any()


def test_flir_thermal_colormaps_and_msx():
    import numpy as np
    from pinn_depth.physics.das_physics import (
        colorize_flir_ironbow,
        colorize_flir_rainbow,
        apply_flir_msx_detail_embossing,
    )

    depth = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    # Ironbow test
    ironbow_rgb = colorize_flir_ironbow(depth)
    assert ironbow_rgb.shape == (10, 10, 3)
    assert ironbow_rgb.dtype == np.uint8
    # Closest item (0.0) should be white-hot
    assert (ironbow_rgb[0, 0] > 240).all()

    # Rainbow test
    rainbow_rgb = colorize_flir_rainbow(depth)
    assert rainbow_rgb.shape == (10, 10, 3)
    assert rainbow_rgb.dtype == np.uint8

    # MSX Embossing test
    dummy_rgb = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    embossed = apply_flir_msx_detail_embossing(ironbow_rgb, dummy_rgb, strength=0.45)
    assert embossed.shape == (10, 10, 3)
    assert embossed.dtype == np.uint8


