"""
Multi-Scale Feature Pyramid Decoder with Residual Refinement
for Physics-Informed Relative Depth Estimation.

Architecture Overview:
    1. Multi-Scale Skip Connections:
       Takes feature maps [C1, C2, C3, C4, C5] from the ResNet encoder:
       - C1: [B,  64, H/2,  W/2]  (Fine spatial details, edges, textures)
       - C2: [B,  64, H/4,  W/4]  (Local object boundaries)
       - C3: [B, 128, H/8,  W/8]  (Mid-level parts & object geometry)
       - C4: [B, 256, H/16, W/16] (High-level semantic regions)
       - C5: [B, 512, H/32, W/32] (Global scene context & sky/horizon)

    2. Top-Down Feature Fusion:
       Starting from the highest semantic feature (C5), features are progressively
       upsampled by 2x using bilinear interpolation, concatenated with projected
       skip features from the corresponding encoder stage, and refined via
       residual convolution blocks.

    3. Mathematical Formulation of Decoder Stage:
       - Skip Projection: S_i = ELU(BatchNorm(Conv1x1(C_i)))
       - Upsample: U_{i+1} = Upsample_2x(D_{i+1})
       - Feature Fusion: F_i = ELU(BatchNorm(Conv3x3(Concat(U_{i+1}, S_i))))
       - Residual Refine: D_i = F_i + Conv3x3(ELU(BatchNorm(Conv3x3(F_i))))
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Refinement Block.
    
    Refines fused feature representations through residual learning,
    preserving high-frequency depth gradients and preventing gradient vanishing:
        y = x + W2 * (ELU(BatchNorm(W1 * x)))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class DecoderStage(nn.Module):
    """
    Individual Feature Pyramid Decoder Stage.
    
    Combines upsampled deep semantic features with lateral skip features
    from earlier encoder layers:
        1. 1x1 conv projects lateral skip features to matching channel dimension.
        2. Bilinear upsampling doubles spatial resolution of deep features.
        3. 3x3 conv fuses the concatenated channels.
        4. ResidualBlock refines spatial details and boundary sharpness.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Lateral skip projection (1x1 conv)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )
        # Feature fusion after bilinear upsampling (3x3 conv)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )
        # Residual refinement block
        self.refine = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Step 1: Lateral projection of high-resolution skip features
        skip_feat = self.skip_proj(skip)
        
        # Step 2: Bilinear upsampling of deep semantic features
        target_size = skip_feat.shape[-2:]
        x_up = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        
        # Step 3: Channel concatenation & fusion
        fused = self.fuse_conv(torch.cat([x_up, skip_feat], dim=1))
        
        # Step 4: Residual boundary refinement
        return self.refine(fused)


class FeaturePyramidDecoder(nn.Module):
    """
    Multi-Scale Feature Pyramid Decoder.
    
    Decodes multi-level encoder features (C1 through C5) into a unified,
    dense 32-channel representation at native image resolution for continuous
    relative depth regression.
    
    Tensor Flow:
        C5 [512, H/32, W/32] -> top_proj    -> [128, H/32, W/32]
        + C4 [256, H/16, W/16] -> stage4    -> [128, H/16, W/16]
        + C3 [128, H/8,  W/8]  -> stage3    -> [128, H/8,  W/8]
        + C2 [64,  H/4,  W/4]  -> stage2    -> [128, H/4,  W/4]
        + C1 [64,  H/2,  W/2]  -> stage1    -> [64,  H/2,  W/2]
        final_stage (upsample to H x W)     -> [32,  H,    W]
    """

    def __init__(self, encoder_channels: List[int] = [64, 64, 128, 256, 512], decoder_channels: int = 64):
        super().__init__()
        self.out_channels = 32
        c1, c2, c3, c4, c5 = encoder_channels

        # Top semantic projection for C5
        self.top_proj = nn.Sequential(
            nn.Conv2d(c5, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            ResidualBlock(128),
        )

        # Progressive decoding stages
        self.stage4 = DecoderStage(in_channels=128, skip_channels=c4, out_channels=128)
        self.stage3 = DecoderStage(in_channels=128, skip_channels=c3, out_channels=128)
        self.stage2 = DecoderStage(in_channels=128, skip_channels=c2, out_channels=128)
        self.stage1 = DecoderStage(in_channels=128, skip_channels=c1, out_channels=64)

        # Final full-resolution refinement stage
        self.final_stage = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            ResidualBlock(32),
        )

    def forward(self, features: List[torch.Tensor], target_size: torch.Size = None) -> torch.Tensor:
        """
        Forward pass decoding multi-scale encoder features.
        
        Args:
            features: [C1, C2, C3, C4, C5] from encoder.
            target_size: Target output spatial resolution (H, W).
            
        Returns:
            Decoded feature map: [B, 32, H, W].
        """
        c1, c2, c3, c4, c5 = features

        # Top projection
        d5 = self.top_proj(c5)

        # Hierarchical decoding with skip connections
        d4 = self.stage4(d5, c4)
        d3 = self.stage3(d4, c3)
        d2 = self.stage2(d3, c2)
        d1 = self.stage1(d2, c1)

        # Upsample to native image size
        if target_size is not None:
            out = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=True)

        return self.final_stage(out)
