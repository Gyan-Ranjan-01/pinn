"""
CNN Encoder Backbones for Relative Depth Estimation.
Supports ResNet-18 and ResNet-50 with pretrained weights and multi-scale feature extraction.
"""

from typing import List, Dict, Optional
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
    Extracts multi-scale feature representations from ResNet backbones.

    Returns feature maps at strides:
        - C1: stride 2
        - C2: stride 4
        - C3: stride 8
        - C4: stride 16
        - C5: stride 32
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            self.channels = [64, 64, 128, 256, 512]
        elif self.backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model = resnet50(weights=weights)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'resnet50'.")

        # Layer breakdown for multi-scale feature taps
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input RGB tensor (B, 3, H, W).

        Returns:
            List of feature tensors [C1, C2, C3, C4, C5].
        """
        # C1: stride 2
        x1 = self.relu(self.bn1(self.conv1(x)))
        # C2: stride 4
        x2 = self.layer1(self.maxpool(x1))
        # C3: stride 8
        x3 = self.layer2(x2)
        # C4: stride 16
        x4 = self.layer3(x3)
        # C5: stride 32
        x5 = self.layer4(x4)

        return [x1, x2, x3, x4, x5]

