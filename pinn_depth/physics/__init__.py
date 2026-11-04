"""
Physics modules for atmospheric scattering, transmission estimation, and consistency losses.
"""

from .scattering import (
    depth_to_transmission,
    transmission_to_depth,
    synthesize_hazy_image,
    recover_radiance,
    AtmosphericScatteringLayer,
)
from .dark_channel import (
    compute_dark_channel_torch,
    estimate_atmospheric_light_torch,
    estimate_transmission_dcp,
)
from .losses import (
    RelativeDepthRankingLoss,
    EdgeAwareSmoothnessLoss,
    ScaleAndShiftInvariantLoss,
    PhysicsConsistencyLoss,
    PINNCombinedLoss,
)
from .das_physics import (
    compute_luminance_das,
    compute_das_attenuation_beta,
    fast_guided_filter,
    get_das_paper_colormap,
    colorize_das_paper,
)

__all__ = [
    "depth_to_transmission",
    "transmission_to_depth",
    "synthesize_hazy_image",
    "recover_radiance",
    "AtmosphericScatteringLayer",
    "compute_dark_channel_torch",
    "estimate_atmospheric_light_torch",
    "estimate_transmission_dcp",
    "RelativeDepthRankingLoss",
    "EdgeAwareSmoothnessLoss",
    "ScaleAndShiftInvariantLoss",
    "PhysicsConsistencyLoss",
    "PINNCombinedLoss",
    "compute_luminance_das",
    "compute_das_attenuation_beta",
    "fast_guided_filter",
    "get_das_paper_colormap",
    "colorize_das_paper",
]

