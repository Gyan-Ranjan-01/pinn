"""
Synthetic Atmospheric Haze Augmentation for Outdoor Depth Training.

Applies Koschmieder's model to clear outdoor images with known depth maps:
    I_hazy(x) = J(x) * exp(-beta * d(x)) + A * (1 - exp(-beta * d(x)))
"""

from typing import Tuple, Optional
import random
import numpy as np
import torch
from pinn_depth.physics.scattering import synthesize_hazy_image


class SyntheticHazeAugmentation:
    """
    Simulates atmospheric haze on the fly during training.
    """

    def __init__(
        self,
        beta_range: Tuple[float, float] = (0.5, 2.5),
        airlight_range: Tuple[float, float] = (0.7, 1.0),
        prob: float = 0.5,
    ):
        self.beta_range = beta_range
        self.airlight_range = airlight_range
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Args:
            image: (3, H, W) float tensor in [0, 1].
            depth: (1, H, W) relative depth float tensor in [0, 1].

        Returns:
            Tuple of:
                augmented_image: (3, H, W)
                transmission: (1, H, W)
                beta: float
                airlight: (3, 1, 1)
        """
        if random.random() > self.prob:
            # No haze applied (clear weather condition)
            ones_t = torch.ones_like(depth)
            airlight = torch.tensor([0.9, 0.9, 0.9], dtype=image.dtype).view(3, 1, 1)
            return image, ones_t, 0.0, airlight

        # Sample physical scattering parameters
        beta = random.uniform(*self.beta_range)
        a_val = random.uniform(*self.airlight_range)

        # Slight color tint in atmospheric light (e.g. blueish or warm daylight)
        tint = torch.tensor(
            [
                a_val * random.uniform(0.95, 1.0),
                a_val * random.uniform(0.95, 1.0),
                a_val * random.uniform(0.98, 1.02),
            ],
            dtype=image.dtype,
        ).view(3, 1, 1)
        airlight = torch.clamp(tint, 0.6, 1.0)

        # Add batch dim for forward pass
        img_b = image.unsqueeze(0)
        dep_b = depth.unsqueeze(0)
        air_b = airlight.unsqueeze(0)

        hazy_b, trans_b = synthesize_hazy_image(img_b, dep_b, airlight=air_b, beta=beta)

        return hazy_b.squeeze(0), trans_b.squeeze(0), beta, airlight

