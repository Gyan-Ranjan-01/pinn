"""
Unit tests for the model architectures.
"""

import pytest
import torch
from pinn_depth.models import PINNDepthNet, ResNetEncoder, FeaturePyramidDecoder


def test_encoder_multiscale_shapes():
    encoder = ResNetEncoder(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    feats = encoder(x)

    assert len(feats) == 5
    # Stride 2, 4, 8, 16, 32
    assert feats[0].shape == (2, 64, 64, 64)
    assert feats[1].shape == (2, 64, 32, 32)
    assert feats[2].shape == (2, 128, 16, 16)
    assert feats[3].shape == (2, 256, 8, 8)
    assert feats[4].shape == (2, 512, 4, 4)


def test_decoder_shape():
    encoder = ResNetEncoder(backbone="resnet18", pretrained=False)
    decoder = FeaturePyramidDecoder(encoder_channels=encoder.channels, decoder_channels=64)
    x = torch.randn(2, 3, 128, 128)
    feats = encoder(x)
    out = decoder(feats, target_size=x.shape[-2:])

    assert out.shape == (2, 32, 128, 128)


def test_pinn_depth_net_forward_and_backward():
    model = PINNDepthNet(backbone="resnet18", pretrained=False, is_physics_informed=True)
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    out = model(x)

    assert "depth" in out
    assert "airlight" in out
    assert "beta" in out
    assert "transmission" in out

    depth = out["depth"]
    assert depth.shape == (2, 1, 64, 64)
    assert (depth >= 0.0).all() and (depth <= 1.0).all()

    # Test backward pass
    loss = depth.sum() + out["airlight"].sum() + out["transmission"].sum()
    loss.backward()
    assert x.grad is not None


def test_baseline_cnn_mode():
    baseline_model = PINNDepthNet(backbone="resnet18", pretrained=False, is_physics_informed=False)
    x = torch.rand(1, 3, 64, 64)
    out = baseline_model(x)

    assert "depth" in out
    assert "airlight" not in out
    assert out["depth"].shape == (1, 1, 64, 64)

