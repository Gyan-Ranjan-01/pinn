"""
Evaluation and Benchmarking Suite for Relative Depth Estimation.
Compares CNN-only baseline vs. Physics-Informed CNN across clear and adverse/hazy scenarios.
"""

from typing import Dict, Any, List, Optional
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from pinn_depth.models.pinn_net import PINNDepthNet
from pinn_depth.dataset.kitti_dataset import OutdoorDepthDataset


def compute_relative_ranking_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    num_pairs: int = 5000,
    margin_tau: float = 0.03,
) -> float:
    """
    Computes pairwise relative depth ordering accuracy (OrdAcc):
    Fraction of randomly sampled pixel pairs whose relative ordering is correctly predicted.
    """
    valid_coords = np.argwhere(mask)
    if len(valid_coords) < 100:
        return 0.0

    n_samples = min(num_pairs, len(valid_coords) // 2)
    idx0 = np.random.choice(len(valid_coords), n_samples, replace=True)
    idx1 = np.random.choice(len(valid_coords), n_samples, replace=True)

    p0 = valid_coords[idx0]
    p1 = valid_coords[idx1]

    gt0 = target[p0[:, 0], p0[:, 1]]
    gt1 = target[p1[:, 0], p1[:, 1]]

    pr0 = pred[p0[:, 0], p0[:, 1]]
    pr1 = pred[p1[:, 0], p1[:, 1]]

    diff_gt = gt0 - gt1
    diff_pr = pr0 - pr1

    # Filter out pairs that are too close in ground truth
    significant = np.abs(diff_gt) > margin_tau
    if not np.any(significant):
        return 1.0

    gt_sign = np.sign(diff_gt[significant])
    pr_sign = np.sign(diff_pr[significant])

    correct = np.sum(gt_sign == pr_sign)
    return float(correct / len(gt_sign))


def compute_standard_depth_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard depth evaluation metrics:
    - AbsRel (Absolute Relative Error)
    - RMSE (Root Mean Squared Error)
    - SILog (Scale-Invariant Logarithmic Error)
    - delta1 (< 1.25 accuracy threshold)
    """
    p = pred[mask]
    t = target[mask]

    if len(p) == 0:
        return {"abs_rel": 0.0, "rmse": 0.0, "silog": 0.0, "delta1": 0.0}

    # Align scale and shift via least squares for fair relative depth evaluation
    # t = s * p + t_shift
    p_mean = np.mean(p)
    t_mean = np.mean(t)
    var_p = np.var(p)
    if var_p > 1e-6:
        scale = np.mean((p - p_mean) * (t - t_mean)) / var_p
        shift = t_mean - scale * p_mean
    else:
        scale = 1.0
        shift = 0.0

    p_aligned = np.clip(scale * p + shift, 1e-3, 100.0)
    t_safe = np.clip(t, 1e-3, 100.0)

    # 1. AbsRel
    abs_rel = float(np.mean(np.abs(p_aligned - t_safe) / t_safe))

    # 2. RMSE
    rmse = float(np.sqrt(np.mean((p_aligned - t_safe) ** 2)))

    # 3. SILog
    d = np.log(p_aligned) - np.log(t_safe)
    silog = float(np.sqrt(np.mean(d ** 2) - (np.mean(d) ** 2)) * 100.0)

    # 4. Threshold delta1 (< 1.25)
    thresh = np.maximum(p_aligned / t_safe, t_safe / p_aligned)
    delta1 = float(np.mean(thresh < 1.25))

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "silog": silog,
        "delta1": delta1,
    }


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_ord_acc: List[float] = []
    all_abs_rel: List[float] = []
    all_rmse: List[float] = []
    all_silog: List[float] = []
    all_delta1: List[float] = []

    for batch in dataloader:
        image = batch["image"].to(device)
        gt_depth = batch["depth"].cpu().numpy()
        mask = batch["valid_mask"].cpu().numpy()

        output = model(image)
        pred_depth = output["depth"].cpu().numpy()

        B = image.shape[0]
        for b in range(B):
            p = pred_depth[b, 0]
            t = gt_depth[b, 0]
            m = mask[b, 0]

            ord_acc = compute_relative_ranking_accuracy(p, t, m)
            std_m = compute_standard_depth_metrics(p, t, m)

            all_ord_acc.append(ord_acc)
            all_abs_rel.append(std_m["abs_rel"])
            all_rmse.append(std_m["rmse"])
            all_silog.append(std_m["silog"])
            all_delta1.append(std_m["delta1"])

    return {
        "ranking_accuracy_ord_acc": float(np.mean(all_ord_acc)),
        "abs_rel": float(np.mean(all_abs_rel)),
        "rmse": float(np.mean(all_rmse)),
        "silog": float(np.mean(all_silog)),
        "delta1_accuracy": float(np.mean(all_delta1)),
    }


def compare_models(
    pinn_checkpoint: str,
    baseline_checkpoint: str,
    dataset_dir: Optional[str] = None,
    backbone: str = "resnet18",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on {device}...")

    # Load test dataset with adverse hazy scenes to test physics robustness
    test_ds = OutdoorDepthDataset(
        root_dir=dataset_dir,
        split="test",
        target_size=(256, 384),
        apply_haze_augmentation=True,
        haze_prob=0.8,
        num_synthetic_fallback=20,
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # 1. Evaluate PINN
    pinn_model = PINNDepthNet(backbone=backbone, pretrained=False, is_physics_informed=True).to(device)
    if os.path.exists(pinn_checkpoint):
        state = torch.load(pinn_checkpoint, map_location=device, weights_only=False)
        pinn_model.load_state_dict(state.get("model_state_dict", state))
        print("Loaded PINN checkpoint.")

    pinn_metrics = benchmark_model(pinn_model, test_loader, device)

    # 2. Evaluate Baseline
    baseline_model = PINNDepthNet(backbone=backbone, pretrained=False, is_physics_informed=False).to(device)
    if os.path.exists(baseline_checkpoint):
        state = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
        baseline_model.load_state_dict(state.get("model_state_dict", state))
        print("Loaded Baseline checkpoint.")

    base_metrics = benchmark_model(baseline_model, test_loader, device)

    print("\n" + "=" * 65)
    print(f"{'Metric':<30} | {'CNN Baseline':<15} | {'PINN (Ours)':<15}")
    print("=" * 65)
    for k in pinn_metrics.keys():
        b_val = base_metrics[k]
        p_val = pinn_metrics[k]
        print(f"{k:<30} | {b_val:<15.4f} | {p_val:<15.4f}")
    print("=" * 65)

    return {"pinn": pinn_metrics, "baseline": base_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PINN vs Baseline")
    parser.add_argument("--pinn-ckpt", type=str, default="checkpoints/best_pinn_model.pt")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/best_baseline_model.pt")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet18")
    args = parser.parse_args()

    compare_models(
        pinn_checkpoint=args.pinn_ckpt,
        baseline_checkpoint=args.baseline_ckpt,
        dataset_dir=args.dataset_dir,
        backbone=args.backbone,
    )

