"""
Physics of Atmospheric Scattering (Koschmieder's Model).

Formulation:
    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

Where:
    - I(x): Observed hazy image at pixel x (B, 3, H, W)
    - J(x): True scene radiance (clear image) (B, 3, H, W)
    - A: Atmospheric light (airlight vector) (B, 3, 1, 1)
    - t(x): Medium transmission map (B, 1, H, W), t in (0, 1]
    - d(x): Scene depth (B, 1, H, W)
    - beta: Atmospheric attenuation / scattering coefficient
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_to_transmission(
    depth: torch.Tensor,
    beta: float | torch.Tensor = 1.0,
    clamp_min: float = 1e-4,
) -> torch.Tensor:
    """
    Computes medium transmission map from scene depth using Koschmieder's law:
        t(x) = exp(-beta * d(x))

    Args:
        depth: (B, 1, H, W) tensor representing relative depth in [0, 1].
        beta: Scattering coefficient (scalar or tensor broadcastable to depth).
        clamp_min: Minimum transmission for numerical stability.

    Returns:
        transmission: (B, 1, H, W) tensor in [clamp_min, 1.0].
    """
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=depth.dtype, device=depth.device)

    # Ensure beta can broadcast to depth
    if beta.dim() < depth.dim():
        while beta.dim() < depth.dim():
            beta = beta.unsqueeze(-1)

    t = torch.exp(-beta * depth)
    return torch.clamp(t, min=clamp_min, max=1.0)


def transmission_to_depth(
    transmission: torch.Tensor,
    beta: float | torch.Tensor = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Inverts Koschmieder's transmission map into relative depth:
        d(x) = - (1 / beta) * ln(t(x))

    Args:
        transmission: (B, 1, H, W) tensor in (0, 1].
        beta: Scattering coefficient.
        eps: Small epsilon to prevent log(0).

    Returns:
        depth: (B, 1, H, W) relative depth tensor.
    """
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=transmission.dtype, device=transmission.device)

    if beta.dim() < transmission.dim():
        while beta.dim() < transmission.dim():
            beta = beta.unsqueeze(-1)

    t_safe = torch.clamp(transmission, min=eps, max=1.0)
    depth = -torch.log(t_safe) / torch.clamp(beta, min=1e-5)
    return depth


def synthesize_hazy_image(
    radiance: torch.Tensor,
    depth: torch.Tensor,
    airlight: Optional[torch.Tensor] = None,
    beta: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesizes a physically consistent hazy image using Koschmieder's model:
        I(x) = J(x) * t(x) + A * (1 - t(x))

    Args:
        radiance: (B, 3, H, W) clear scene radiance in [0, 1].
        depth: (B, 1, H, W) relative depth in [0, 1].
        airlight: (B, 3, 1, 1) atmospheric light. If None, default bright sky [0.95, 0.95, 0.95].
        beta: Scattering coefficient.

    Returns:
        Tuple of:
            hazy_image: (B, 3, H, W) simulated hazy image in [0, 1].
            transmission: (B, 1, H, W) transmission map in [0, 1].
    """
    if airlight is None:
        airlight = torch.tensor([0.95, 0.95, 0.95], dtype=radiance.dtype, device=radiance.device)
        airlight = airlight.view(1, 3, 1, 1)

    # Ensure airlight shape is (B, 3, 1, 1)
    if airlight.dim() == 1:
        airlight = airlight.view(1, -1, 1, 1)
    elif airlight.dim() == 2:
        airlight = airlight.view(-1, 3, 1, 1)

    t = depth_to_transmission(depth, beta=beta)
    hazy = radiance * t + airlight * (1.0 - t)
    return torch.clamp(hazy, 0.0, 1.0), t


def recover_radiance(
    hazy: torch.Tensor,
    transmission: torch.Tensor,
    airlight: torch.Tensor,
    t_min: float = 0.1,
) -> torch.Tensor:
    """
    Recovers scene radiance (dehazing inversion) from hazy image:
        J(x) = (I(x) - A) / max(t(x), t_min) + A

    Args:
        hazy: (B, 3, H, W) hazy image in [0, 1].
        transmission: (B, 1, H, W) transmission in (0, 1].
        airlight: (B, 3, 1, 1) atmospheric light.
        t_min: Lower bound threshold for transmission to prevent division by near-zero.

    Returns:
        recovered_radiance: (B, 3, H, W) reconstructed radiance in [0, 1].
    """
    if airlight.dim() == 1:
        airlight = airlight.view(1, -1, 1, 1)
    elif airlight.dim() == 2:
        airlight = airlight.view(-1, 3, 1, 1)

    t_clamped = torch.clamp(transmission, min=t_min, max=1.0)
    radiance = (hazy - airlight) / t_clamped + airlight
    return torch.clamp(radiance, 0.0, 1.0)


class AtmosphericScatteringLayer(nn.Module):
    """
    Differentiable Atmospheric Scattering Layer.
    Allows end-to-end backpropagation through the physics simulation.
    """

    def __init__(self, default_beta: float = 1.0, learnable_beta: bool = False):
        super().__init__()
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor([default_beta], dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor([default_beta], dtype=torch.float32))

    def forward(
        self,
        radiance: torch.Tensor,
        depth: torch.Tensor,
        airlight: Optional[torch.Tensor] = None,
        custom_beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = custom_beta if custom_beta is not None else self.beta
        return synthesize_hazy_image(radiance, depth, airlight=airlight, beta=b)

    def recover(
        self,
        hazy: torch.Tensor,
        depth: torch.Tensor,
        airlight: torch.Tensor,
        custom_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = custom_beta if custom_beta is not None else self.beta
        t = depth_to_transmission(depth, beta=b)
        return recover_radiance(hazy, t, airlight)

