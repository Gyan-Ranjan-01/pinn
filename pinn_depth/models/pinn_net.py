"""
=============================================================================
PINN-Depth: Physics-Informed Relative Depth Estimation Network
=============================================================================
This is the core neural network combining deep visual feature extraction with
Koschmieder's physical atmospheric scattering constraint.

Core Idea (How to explain this project):
    1. Vision Model:
       An RGB image goes through a ResNet-18 Encoder and an FPN Decoder.
       The network predicts a continuous relative depth map d(x) in [0.0, 1.0].
       - 0.0 represents the nearest foreground objects (warm/yellow).
       - 1.0 represents the farthest objects or sky horizon (cool/dark).

    2. Physics Model (Koschmieder's Atmospheric Law):
       In outdoor scenes, atmospheric haze accumulates with distance:
           I(x) = J(x) * t(x) + A * (1 - t(x))
           t(x) = exp(-beta * d(x))
       Where:
           - I(x): The input outdoor image observed by the camera.
           - J(x): The clear unscattered physical scene radiance.
           - A: The atmospheric airlight (sky light vector).
           - beta: Atmospheric attenuation coefficient.
           - t(x): Transmission map (how much clear light reaches the camera).
           - d(x): Relative distance of each object.

    3. Physics-Informed Optimization:
       Instead of needing expensive LiDAR sensors, the network learns depth by
       enforcing that its predicted depth d(x) and estimated airlight A obey the
       physical laws of light scattering in outdoor atmospheres.
=============================================================================
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNetEncoder
from .decoder import FeaturePyramidDecoder


class RelativeDepthHead(nn.Module):
    """
    Relative Depth Prediction Head.
    Maps decoded features [Batch, 32, H, W] to a single-channel depth map
    [Batch, 1, H, W] bounded strictly between 0.0 and 1.0 using a Sigmoid.
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        # 3x3 Conv + ELU for smooth spatial transitions
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True)
        self.act = nn.ELU(inplace=True)

        # 1x1 Conv to project 16 channels -> 1 depth channel
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1, bias=True)

        # Sigmoid guarantees relative depth is bounded in [0.0, 1.0]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [Batch, 32, Height, Width]
        Output: [Batch,  1, Height, Width] in [0.0, 1.0]
        """
        x = self.act(self.conv1(x))
        depth = self.sigmoid(self.conv2(x))
        return depth


class AtmosphericPhysicsHead(nn.Module):
    """
    Atmospheric Parameter Estimator.
    Estimates the global atmospheric Airlight vector 'A' (RGB) and the
    extinction coefficient 'beta' from the top global scene feature (C5).
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()
        # Global pooling collapses [H/32, W/32] into a single 1x1 global summary vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Small 2-layer MLP to predict 4 scalar physical parameters
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 4),  # Output: [A_red, A_green, A_blue, beta]
        )

    def forward(self, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            c5: Deepest encoder feature map [Batch, 512, H/32, W/32].
        Returns:
            airlight: Vector A bounded in [0.5, 1.0] of shape [Batch, 3, 1, 1].
            beta:     Attenuation coefficient bounded in [0.2, 3.0] of shape [Batch, 1, 1, 1].
        """
        # Collapse spatial dimensions to 1D feature vector: [Batch, 512]
        feat = self.pool(c5).flatten(1)
        params = self.fc(feat)

        # Airlight A is bounded between [0.5, 1.0] (sky/haze is bright outdoor light)
        airlight = 0.5 + 0.5 * torch.sigmoid(params[:, :3]).view(-1, 3, 1, 1)

        # Beta is bounded between [0.2, 3.0] (natural atmospheric scattering range)
        beta = 0.2 + 2.8 * torch.sigmoid(params[:, 3:]).view(-1, 1, 1, 1)

        return airlight, beta


class DifferentiableAtmosphericLayer(nn.Module):
    """
    Differentiable Koschmieder Scattering Layer.
    Converts predicted depth d(x) into physical atmospheric transmission t(x),
    and recovers the physical unscattered scene radiance J(x).
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def depth_to_transmission(self, depth: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Koschmieder Equation:
            t(x) = exp(-beta * d(x))
        Near objects (d -> 0): t -> 1.0 (clear transmission)
        Far objects  (d -> 1): t -> exp(-beta) (attenuated by atmospheric haze)
        """
        transmission = torch.exp(-beta * depth)
        return torch.clamp(transmission, min=self.eps, max=1.0)

    def recover_radiance(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        airlight: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Physical Dehazing Equation:
            J(x) = (I(x) - A) / max(t(x), eps) + A
        Recovers the intrinsic true surface radiance without atmospheric scattering.
        """
        t = self.depth_to_transmission(depth, beta)
        j = (image - airlight) / torch.clamp(t, min=self.eps) + airlight
        return torch.clamp(j, 0.0, 1.0)


class PINNDepthNet(nn.Module):
    """
    Unified Physics-Informed Relative Monocular Depth Network.

    How Data Flows Through the Model:
        1. Input Image [Batch, 3, H, W]
        2. ResNetEncoder -> extracts 5 multi-scale feature maps [C1, C2, C3, C4, C5]
        3. FeaturePyramidDecoder -> fuses deep context with fine edges -> [Batch, 32, H, W]
        4. RelativeDepthHead -> predicts bounded continuous relative depth [Batch, 1, H, W]
        5. AtmosphericPhysicsHead -> predicts global airlight A and scattering beta
        6. DifferentiableAtmosphericLayer -> computes physical transmission t(x) and radiance J(x)
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

        # 1. Visual Feature Extractor
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)

        # 2. Multi-Scale Decoder (Fuses C1..C5)
        self.decoder = FeaturePyramidDecoder(encoder_channels=self.encoder.channels)

        # 3. Relative Depth Head
        self.depth_head = RelativeDepthHead(in_channels=self.decoder.out_channels)

        # 4. Atmospheric Physics Components
        if self.is_physics_informed:
            self.physics_head = AtmosphericPhysicsHead(in_channels=self.encoder.channels[-1])
            self.scattering_layer = DifferentiableAtmosphericLayer()
        else:
            self.physics_head = None
            self.scattering_layer = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass for Relative Depth Estimation.

        Args:
            x: Input RGB tensor [Batch, 3, Height, Width] in [0.0, 1.0].

        Returns:
            Dictionary containing:
                - 'depth': Predicted relative depth map [Batch, 1, H, W] in [0, 1]
                - 'airlight': Estimated airlight vector A [Batch, 3, 1, 1] (if physics enabled)
                - 'beta': Estimated scattering factor beta [Batch, 1, 1, 1] (if physics enabled)
                - 'transmission': Koschmieder transmission map t(x) [Batch, 1, H, W]
                - 'recovered_radiance': Clear physical scene radiance J(x) [Batch, 3, H, W]
        """
        target_size = x.shape[-2:]

        # --- Step 1: Extract multi-scale visual features ---
        features = self.encoder(x)  # [C1, C2, C3, C4, C5]

        # --- Step 2: Decode into high-resolution 32-channel feature map ---
        decoded = self.decoder(features, target_size=target_size)

        # --- Step 3: Predict bounded relative depth in [0.0, 1.0] ---
        depth = self.depth_head(decoded)

        result: Dict[str, torch.Tensor] = {
            "depth": depth,
        }

        # --- Step 4: Physical atmospheric constraint evaluation ---
        if self.is_physics_informed and self.physics_head is not None:
            # Predict global atmospheric parameters from deepest semantic feature C5
            airlight, beta = self.physics_head(features[-1])

            # Compute physical atmospheric transmission: t(x) = exp(-beta * depth)
            transmission = self.scattering_layer.depth_to_transmission(depth, beta)

            # Recover unscattered scene radiance: J(x)
            recovered_radiance = self.scattering_layer.recover_radiance(x, depth, airlight, beta)

            result["airlight"] = airlight
            result["beta"] = beta
            result["transmission"] = transmission
            result["recovered_radiance"] = recovered_radiance

        return result
