"""
=============================================================================
PINN-Depth: Multi-Scale Feature Pyramid Decoder with Residual Refinement
=============================================================================
This module combines deep semantic scene features (C5) with sharp shallow edge
features (C1, C2, C3, C4) to construct a high-resolution, edge-preserving
depth representation.

How It Works (Top-Down Progressive Fusion):
    1. Start at the highest semantic feature: C5 [512 channels, H/32, W/32].
    2. At each stage:
       - Upsample the current deep feature by 2x using Bilinear Interpolation.
       - Project the lateral skip feature from the encoder using a 1x1 Conv.
       - Concatenate them along the channel dimension.
       - Fuse channels with a 3x3 Conv + ELU + BatchNorm.
       - Refine boundaries with a Residual Block: y = x + F(x).
    3. Final Stage:
       - Upsample to native image resolution (H x W) and output 32 clean channels.

Tensor Dimensions Progression:
    - Input: [C1, C2, C3, C4, C5] from ResNetEncoder
    - Stage 4: Fuses (Upsample(C5), C4) -> [128 channels, H/16, W/16]
    - Stage 3: Fuses (Upsample(D4), C3) -> [128 channels, H/8,  W/8 ]
    - Stage 2: Fuses (Upsample(D3), C2) -> [128 channels, H/4,  W/4 ]
    - Stage 1: Fuses (Upsample(D2), C1) -> [ 64 channels, H/2,  W/2 ]
    - Final:   Upsample(D1) -> Conv3x3  -> [ 32 channels, H,    W   ]
=============================================================================
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Refinement Block.
    Refines feature representations without gradient degradation:
        output = input + Conv3x3(ELU(BN(Conv3x3(ELU(BN(input))))))
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
    Individual FPN Decoder Stage.
    Fuses deep semantic features with shallow high-resolution skip features.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # 1x1 Conv to match skip connection channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

        # 3x3 Conv to fuse upsampled deep features + projected skip features
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

        # Residual block to refine object boundaries
        self.refine = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Step 1: Project lateral skip connection
        skip_feat = self.skip_proj(skip)

        # Step 2: Upsample deep feature by 2x to match skip feature spatial size
        target_size = skip_feat.shape[-2:]
        x_up = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)

        # Step 3: Concatenate and fuse
        fused = self.fuse_conv(torch.cat([x_up, skip_feat], dim=1))

        # Step 4: Refine boundaries
        return self.refine(fused)


class FeaturePyramidDecoder(nn.Module):
    """
    Hierarchical Feature Pyramid Decoder.
    Combines C1 through C5 into a 32-channel feature map at native image resolution.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        decoder_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = 32
        c1, c2, c3, c4, c5 = encoder_channels

        # Top projection: Compress C5 (512 channels) to 128 channels
        self.top_proj = nn.Sequential(
            nn.Conv2d(c5, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            ResidualBlock(128),
        )

        # Top-down decoding stages (Stride 32 -> 16 -> 8 -> 4 -> 2)
        self.stage4 = DecoderStage(in_channels=128, skip_channels=c4, out_channels=128)
        self.stage3 = DecoderStage(in_channels=128, skip_channels=c3, out_channels=128)
        self.stage2 = DecoderStage(in_channels=128, skip_channels=c2, out_channels=128)
        self.stage1 = DecoderStage(in_channels=128, skip_channels=c1, out_channels=64)

        # Final stage: Upsample to native image size (H x W) and reduce to 32 channels
        self.final_stage = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            ResidualBlock(32),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [C1, C2, C3, C4, C5] from ResNetEncoder.
            target_size: Target spatial dimensions (H, W).
        Returns:
            Unified feature representation of shape [Batch, 32, Height, Width].
        """
        c1, c2, c3, c4, c5 = features

        # Step 1: Compress top semantic feature (C5)
        d5 = self.top_proj(c5)

        # Step 2: Progressively fuse with skip connections
        d4 = self.stage4(d5, c4)  # Resolution: H/16 x W/16
        d3 = self.stage3(d4, c3)  # Resolution: H/8  x W/8
        d2 = self.stage2(d3, c2)  # Resolution: H/4  x W/4
        d1 = self.stage1(d2, c1)  # Resolution: H/2  x W/2

        # Step 3: Final upsampling to original input resolution
        if target_size is not None:
            out = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=True)

        return self.final_stage(out)
