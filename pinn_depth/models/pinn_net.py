"""
Physics-Informed Neural Network (PINN) Architecture for Relative Depth Estimation.

Mathematical Foundation:
    1. Visual Perception Model (CNN + ASPP + Attention FPN):
       Extracts dense multi-scale visual cues across diverse outdoor scenes.
    2. Koschmieder's Atmospheric Scattering Law:
       I(x) = J(x) * t(x) + A * (1 - t(x))
       t(x) = exp(-beta * d(x))
    3. Inverted Physical Depth Constraint:
       d(x) = - (1 / beta) * ln(t(x))

Where:
    - I(x): Observed hazy outdoor image (RGB)
    - J(x): True unscattered surface radiance (clear image)
    - A: Atmospheric light vector (airlight)
    - t(x): Medium transmission map in (0, 1]
    - d(x): Continuous relative depth map in [0, 1] (0 = closest, 1 = farthest / sky)
    - beta: Atmospheric attenuation coefficient
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNetEncoder
from .decoder import FeaturePyramidDecoder


class RelativeDepthHead(nn.Module):
    """
    Continuous Relative Depth Prediction Head.
    Maps decoded multi-scale feature representations to a continuous relative
    depth map normalized strictly to [0.0, 1.0].

    Design:
        - 3x3 Conv + ELU activation (boundary-preserving spatial smoothing)
        - 1x1 Projection Conv
        - Sigmoid activation ensuring bounded relative depth:
          d(x) -> 0.0: Nearest foreground object
          d(x) -> 1.0: Farthest background object / sky horizon
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True)
        self.act = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, in_channels, H, W] -> Output: [B, 1, H, W] in [0, 1]
        out = self.act(self.conv1(x))
        depth = self.sigmoid(self.conv2(out))
        return depth


class AtmosphericPhysicsHead(nn.Module):
    """
    Global Atmospheric Parameter Estimator.
    Estimates physical airlight vector 'A' and attenuation coefficient 'beta'
    directly from global scene context (C5 feature representation).

    Physical Ranges:
        - Airlight A in [0.5, 1.0]^3: Outdoor atmospheric sky and haze illumination.
        - Beta in [0.2, 3.0]: Aerosol scattering coefficient in natural atmospheres.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 4),  # 3 channels for airlight (R, G, B), 1 channel for beta
        )

    def forward(self, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: C5 feature map [B, in_channels, H_c5, W_c5]
        feat = self.pool(c5).flatten(1)
        params = self.fc(feat)

        # Bounded Airlight Vector A in [0.5, 1.0]
        airlight = 0.5 + 0.5 * torch.sigmoid(params[:, :3]).view(-1, 3, 1, 1)

        # Bounded Attenuation Coefficient beta in [0.2, 3.0]
        beta = 0.2 + 2.8 * torch.sigmoid(params[:, 3:]).view(-1, 1, 1, 1)

        return airlight, beta


class DifferentiableAtmosphericLayer(nn.Module):
    """
    Differentiable Atmospheric Scattering Layer implementing Koschmieder's physical equations.
    Allows end-to-end gradient backpropagation through physical constraints.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def depth_to_transmission(self, depth: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Koschmieder transmission: t(x) = exp(-beta * d(x))"""
        t = torch.exp(-beta * depth)
        return torch.clamp(t, min=self.eps, max=1.0)

    def recover_radiance(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        airlight: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Physical scene radiance recovery (dehazing): J(x) = (I(x) - A) / max(t(x), eps) + A"""
        t = self.depth_to_transmission(depth, beta)
        j = (image - airlight) / torch.clamp(t, min=self.eps) + airlight
        return torch.clamp(j, 0.0, 1.0)


class PINNDepthNet(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Relative Monocular Depth Estimation.

    Forward Pass Workflow:
        1. RGB Input -> Multi-scale ResNet Feature Hierarchy [C1, C2, C3, C4, C5]
        2. Contextual Scene Modeling -> Atrous Spatial Pyramid Pooling (ASPP)
        3. Multi-Scale Decoder -> Additive Attention Gates + Residual Refinement
        4. Relative Depth Head -> Dense Relative Depth d(x) in [0.0, 1.0]
        5. Atmospheric Physics Head -> Estimates global Airlight A and Attenuation beta
        6. Differentiable Scattering Layer -> Computes Transmission t(x) = exp(-beta * d(x))
           and unscattered physical radiance J(x).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        is_physics_informed: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.is_physics_informed = is_physics_informed

        # 1. Multi-scale feature extractor
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)

        # 2. Multi-scale ASPP + Attention-gated decoder
        self.decoder = FeaturePyramidDecoder(encoder_channels=self.encoder.channels)

        # 3. Continuous relative depth head
        self.depth_head = RelativeDepthHead(in_channels=self.decoder.out_channels)

        # 4. Atmospheric scattering physics components
        if self.is_physics_informed:
            self.physics_head = AtmosphericPhysicsHead(in_channels=self.encoder.channels[-1])
            self.scattering_layer = DifferentiableAtmosphericLayer()
        else:
            self.physics_head = None
            self.scattering_layer = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for relative depth estimation.

        Args:
            x: Input RGB image tensor of shape (B, 3, H, W) normalized to [0, 1].

        Returns:
            Dictionary containing:
                - 'depth': Predicted relative depth map (B, 1, H, W) in [0, 1]
                - 'transmission': Physical atmospheric transmission map (B, 1, H, W)
                - 'airlight': Estimated atmospheric illuminance vector (B, 3, 1, 1)
                - 'beta': Estimated scattering coefficient (B, 1, 1, 1)
                - 'recovered_radiance': Physical unscattered radiance J(x) (B, 3, H, W)
        """
        target_size = x.shape[-2:]

        # Step 1: Multi-scale visual feature extraction
        # features = [C1 (H/2), C2 (H/4), C3 (H/8), C4 (H/16), C5 (H/32)]
        features = self.encoder(x)

        # Step 2: ASPP multi-scale context & Attention-Gated FPN decoding
        decoded_features = self.decoder(features, target_size=target_size)

        # Step 3: Relative depth prediction
        pred_depth = self.depth_head(decoded_features)

        result: Dict[str, torch.Tensor] = {
            "depth": pred_depth,
        }

        # Step 4: Physics-informed atmospheric parameter estimation
        if self.is_physics_informed and self.physics_head is not None:
            c5 = features[-1]
            airlight, beta = self.physics_head(c5)

            # Step 5: Differentiable Koschmieder transmission: t(x) = exp(-beta * d(x))
            transmission = self.scattering_layer.depth_to_transmission(pred_depth, beta)

            # Step 6: Physical unscattered scene radiance: J(x)
            recovered_radiance = self.scattering_layer.recover_radiance(x, pred_depth, airlight, beta)

            result["airlight"] = airlight
            result["beta"] = beta
            result["transmission"] = transmission
            result["recovered_radiance"] = recovered_radiance

        return result
