"""
Loss Functions for Physics-Informed Relative Depth Estimation.

Includes:
    - RelativeDepthRankingLoss: Ordinal pairwise ranking loss
    - EdgeAwareSmoothnessLoss: Edge-guided gradient smoothness
    - ScaleAndShiftInvariantLoss: Scale-and-shift invariant relative depth loss
    - PhysicsConsistencyLoss: Atmospheric scattering & transmission consistency
    - PINNCombinedLoss: Unified multi-task objective
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scattering import depth_to_transmission, recover_radiance
from .dark_channel import estimate_transmission_dcp, compute_dark_channel_torch


class RelativeDepthRankingLoss(nn.Module):
    """
    Pairwise Ordinal Ranking Loss for relative depth.
    Samples random pairs of pixels and penalizes depth ordering violations.
    """

    def __init__(self, num_pairs: int = 5000, margin_tau: float = 0.03):
        super().__init__()
        self.num_pairs = num_pairs
        self.margin_tau = margin_tau

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: (B, 1, H, W) predicted relative depth in [0, 1].
            gt_depth: (B, 1, H, W) ground truth depth (relative or metric).
            valid_mask: (B, 1, H, W) boolean mask of valid pixels.

        Returns:
            Scalar ranking loss.
        """
        B, _, H, W = pred_depth.shape
        total_loss = torch.tensor(0.0, device=pred_depth.device)

        if valid_mask is None:
            valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)

        valid_count = 0

        for b in range(B):
            mask_b = valid_mask[b, 0]
            valid_indices = torch.nonzero(mask_b, as_tuple=False)

            num_valid = valid_indices.shape[0]
            if num_valid < 100:
                continue

            n_samples = min(self.num_pairs, num_valid // 2)

            idx0 = torch.randint(0, num_valid, (n_samples,), device=pred_depth.device)
            idx1 = torch.randint(0, num_valid, (n_samples,), device=pred_depth.device)

            p0 = valid_indices[idx0]  # (n_samples, 2)
            p1 = valid_indices[idx1]

            # Ground-truth values
            gt0 = gt_depth[b, 0, p0[:, 0], p0[:, 1]]
            gt1 = gt_depth[b, 0, p1[:, 0], p1[:, 1]]

            # Predicted values
            pr0 = pred_depth[b, 0, p0[:, 0], p0[:, 1]]
            pr1 = pred_depth[b, 0, p1[:, 0], p1[:, 1]]

            diff_gt = gt0 - gt1
            diff_pr = pr0 - pr1

            # Determine ordinal labels: +1 (gt0 > gt1), -1 (gt0 < gt1), 0 (roughly equal)
            labels = torch.zeros_like(diff_gt)
            labels[diff_gt > self.margin_tau] = 1.0
            labels[diff_gt < -self.margin_tau] = -1.0

            # Inequality pairs: smooth margin ranking loss log(1 + exp(-l * diff))
            ineq_mask = labels != 0
            if ineq_mask.any():
                loss_ineq = torch.log1p(torch.exp(-labels[ineq_mask] * diff_pr[ineq_mask]))
                total_loss = total_loss + loss_ineq.mean()

            # Equality pairs: (pr0 - pr1)^2
            eq_mask = labels == 0
            if eq_mask.any():
                loss_eq = torch.pow(diff_pr[eq_mask], 2)
                total_loss = total_loss + loss_eq.mean()

            valid_count += 1

        if valid_count > 0:
            return total_loss / valid_count
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge-Aware Smoothness Loss.
    Penalizes high gradients in predicted depth, weighted by inverse image gradient.
    Preserves true depth discontinuities at object boundaries.
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_depth: (B, 1, H, W) depth map.
            image: (B, 3, H, W) RGB image in [0, 1].

        Returns:
            Scalar smoothness loss.
        """
        # Gradients of depth
        d_dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
        d_dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])

        # Gradients of image (mean across color channels)
        img_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        img_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

        # Exponential edge weights
        weight_x = torch.exp(-self.alpha * img_dx)
        weight_y = torch.exp(-self.alpha * img_dy)

        smooth_loss = torch.mean(d_dx * weight_x) + torch.mean(d_dy * weight_y)
        return smooth_loss


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale-and-Shift Invariant Loss for relative depth supervision.
    Normalizes both prediction and target by their median and mean absolute deviation.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = (target > 0) & torch.isfinite(target)

        B = pred.shape[0]
        loss = torch.tensor(0.0, device=pred.device)
        valid_b = 0

        for b in range(B):
            m_b = mask[b, 0]
            if not m_b.any():
                continue

            p_vals = pred[b, 0][m_b]
            t_vals = target[b, 0][m_b]

            # Median normalization
            p_med = torch.median(p_vals)
            t_med = torch.median(t_vals)

            p_mad = torch.mean(torch.abs(p_vals - p_med)) + self.eps
            t_mad = torch.mean(torch.abs(t_vals - t_med)) + self.eps

            p_norm = (p_vals - p_med) / p_mad
            t_norm = (t_vals - t_med) / t_mad

            loss = loss + torch.mean(torch.abs(p_norm - t_norm))
            valid_b += 1

        if valid_b > 0:
            return loss / valid_b
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics Consistency Loss based on atmospheric scattering principles:
    1. Transmission map consistency between network relative depth and DCP transmission.
    2. Scene radiance reconstruction consistency (clear scene dark channel minimization).
    3. Haze reconstruction loss on synthesized hazy inputs.
    """

    def __init__(
        self,
        lambda_trans: float = 1.0,
        lambda_dark: float = 0.5,
        default_beta: float = 1.2,
    ):
        super().__init__()
        self.lambda_trans = lambda_trans
        self.lambda_dark = lambda_dark
        self.default_beta = default_beta

    def forward(
        self,
        pred_depth: torch.Tensor,
        image: torch.Tensor,
        airlight: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_depth: (B, 1, H, W) relative depth in [0, 1].
            image: (B, 3, H, W) outdoor image in [0, 1].
            airlight: Optional atmospheric light (B, 3, 1, 1).
            beta: Optional scattering coefficient.

        Returns:
            Dictionary with 'loss_total', 'loss_trans', and 'loss_dark'.
        """
        b_val = beta if beta is not None else self.default_beta

        # 1. Compute predicted physical transmission
        pred_trans = depth_to_transmission(pred_depth, beta=b_val)

        # 2. Compute empirical transmission via Dark Channel Prior
        dcp_trans, est_airlight = estimate_transmission_dcp(image, airlight=airlight)

        # Transmission consistency loss
        loss_trans = F.l1_loss(pred_trans, dcp_trans.detach())

        # 3. Recover scene radiance and minimize residual dark channel
        rec_radiance = recover_radiance(image, pred_trans, est_airlight)
        dark_ch = compute_dark_channel_torch(rec_radiance, patch_size=15)
        loss_dark = torch.mean(dark_ch)

        total_phys = self.lambda_trans * loss_trans + self.lambda_dark * loss_dark

        return {
            "loss_total": total_phys,
            "loss_trans": loss_trans,
            "loss_dark": loss_dark,
            "pred_trans": pred_trans,
            "dcp_trans": dcp_trans,
            "airlight": est_airlight,
        }


class PINNCombinedLoss(nn.Module):
    """
    Unified multi-task loss for training the Physics-Informed Relative Depth Model.
    Combines:
        L_total = w_rank * L_rank + w_phys * L_phys + w_ssi * L_ssi + w_smooth * L_smooth
    """

    def __init__(
        self,
        w_rank: float = 1.0,
        w_phys: float = 0.5,
        w_ssi: float = 1.0,
        w_smooth: float = 0.1,
        is_physics_informed: bool = True,
    ):
        super().__init__()
        self.w_rank = w_rank
        self.w_phys = w_phys
        self.w_ssi = w_ssi
        self.w_smooth = w_smooth
        self.is_physics_informed = is_physics_informed

        self.ranking_loss = RelativeDepthRankingLoss()
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
        self.ssi_loss = ScaleAndShiftInvariantLoss()
        self.physics_loss = PhysicsConsistencyLoss()

    def forward(
        self,
        pred_depth: torch.Tensor,
        image: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        airlight: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=pred_depth.device)

        # Smoothness loss
        l_smooth = self.smoothness_loss(pred_depth, image)
        losses["smooth"] = l_smooth
        total = total + self.w_smooth * l_smooth

        # Supervision if ground truth is available
        if gt_depth is not None:
            l_ssi = self.ssi_loss(pred_depth, gt_depth, valid_mask)
            l_rank = self.ranking_loss(pred_depth, gt_depth, valid_mask)
            losses["ssi"] = l_ssi
            losses["rank"] = l_rank
            total = total + self.w_ssi * l_ssi + self.w_rank * l_rank

        # Physics-informed consistency
        if self.is_physics_informed:
            phys_dict = self.physics_loss(pred_depth, image, airlight=airlight, beta=beta)
            losses["phys"] = phys_dict["loss_total"]
            losses["phys_trans"] = phys_dict["loss_trans"]
            losses["phys_dark"] = phys_dict["loss_dark"]
            total = total + self.w_phys * phys_dict["loss_total"]

        losses["total"] = total
        return losses

