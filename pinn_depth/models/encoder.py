"""
=============================================================================
PINN-Depth: Multi-Scale Visual Feature Encoder
=============================================================================
This module extracts visual features at 5 different spatial scales using a
standard ResNet backbone (ResNet-18 by default).

Why Multi-Scale?
    - Shallow layers (C1, C2) preserve sharp spatial edges, textures, and object boundaries.
    - Deep layers (C4, C5) understand high-level semantics, scene geometry, and the horizon.

Feature Hierarchy (for an input image of size H x W):
    - C1: [Batch,  64, H/2,  W/2 ] -> Stride 2  (Fine edges, leaf contours, surface textures)
    - C2: [Batch,  64, H/4,  W/4 ] -> Stride 4  (Object boundaries and local shapes)
    - C3: [Batch, 128, H/8,  W/8 ] -> Stride 8  (Object parts and mid-level structures)
    - C4: [Batch, 256, H/16, W/16] -> Stride 16 (Semantic regions: road, trees, buildings)
    - C5: [Batch, 512, H/32, W/32] -> Stride 32 (Global context: sky, horizon, atmospheric haze)
=============================================================================
"""

from typing import List
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)


class ResNetEncoder(nn.Module):
    """
    Extracts multi-scale features from an input RGB image using a ResNet backbone.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        # Load standard torchvision ResNet weights
        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            # Channel counts at each of the 5 stages
            self.channels = [64, 64, 128, 256, 512]
        elif self.backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model = resnet50(weights=weights)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: '{backbone}'. Please choose 'resnet18' or 'resnet50'.")

        # Stage 1: Initial 7x7 Convolution + MaxPool
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Stages 2 - 5: Residual Blocks
        self.layer1 = base_model.layer1  # Stride 4
        self.layer2 = base_model.layer2  # Stride 8
        self.layer3 = base_model.layer3  # Stride 16
        self.layer4 = base_model.layer4  # Stride 32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass: extracts feature representations at 5 progressive scales.

        Args:
            x: Input RGB tensor [Batch, 3, Height, Width] normalized to [0, 1].

        Returns:
            List of 5 feature tensors: [C1, C2, C3, C4, C5]
        """
        # --- Stage C1 (Stride 2): Initial 7x7 conv captures raw edge gradients ---
        c1 = self.relu(self.bn1(self.conv1(x)))

        # --- Stage C2 (Stride 4): MaxPool + ResNet Layer 1 ---
        x_pool = self.maxpool(c1)
        c2 = self.layer1(x_pool)

        # --- Stage C3 (Stride 8): ResNet Layer 2 ---
        c3 = self.layer2(c2)

        # --- Stage C4 (Stride 16): ResNet Layer 3 ---
        c4 = self.layer3(c3)

        # --- Stage C5 (Stride 32): ResNet Layer 4 (Global Scene Representation) ---
        c5 = self.layer4(c4)

        return [c1, c2, c3, c4, c5]
