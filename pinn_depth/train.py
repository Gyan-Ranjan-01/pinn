"""
Training Engine for Physics-Informed Relative Depth Estimation.
Supports training both the Physics-Informed model and CNN-only baseline for comparative ablation.
"""

import os
import json
import argparse
import time
from typing import Optional, Dict, List
import torch
from torch.utils.data import DataLoader
import numpy as np

from pinn_depth.models.pinn_net import PINNDepthNet
from pinn_depth.physics.losses import PINNCombinedLoss
from pinn_depth.dataset.kitti_dataset import OutdoorDepthDataset


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_physics: bool = True,
) -> Dict[str, float]:
    model.train()
    running_losses: Dict[str, float] = {}
    total_batches = 0

    for batch in dataloader:
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        mask = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        output = model(image)
        pred_depth = output["depth"]
        airlight = output.get("airlight", None)
        beta = output.get("beta", None)

        loss_dict = criterion(
            pred_depth=pred_depth,
            image=image,
            gt_depth=gt_depth,
            valid_mask=mask,
            airlight=airlight,
            beta=beta,
        )

        loss = loss_dict["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_batches += 1
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

    return {k: v / max(1, total_batches) for k, v in running_losses.items()}


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    val_losses: Dict[str, float] = {}
    total_batches = 0

    for batch in dataloader:
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        mask = batch["valid_mask"].to(device)

        output = model(image)
        pred_depth = output["depth"]
        airlight = output.get("airlight", None)
        beta = output.get("beta", None)

        loss_dict = criterion(
            pred_depth=pred_depth,
            image=image,
            gt_depth=gt_depth,
            valid_mask=mask,
            airlight=airlight,
            beta=beta,
        )

        total_batches += 1
        for k, v in loss_dict.items():
            val_losses[k] = val_losses.get(k, 0.0) + v.item()

    return {k: v / max(1, total_batches) for k, v in val_losses.items()}


def train(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    backbone: str = "resnet18",
    is_physics_informed: bool = True,
    dataset_dir: Optional[str] = None,
    save_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Physics-Informed: {is_physics_informed} | Backbone: {backbone}")

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_name = "best_pinn_model.pt" if is_physics_informed else "best_baseline_model.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    train_ds = OutdoorDepthDataset(
        root_dir=dataset_dir,
        split="train",
        target_size=(256, 384),
        apply_haze_augmentation=is_physics_informed,
        num_synthetic_fallback=40,
    )
    val_ds = OutdoorDepthDataset(
        root_dir=dataset_dir,
        split="val",
        target_size=(256, 384),
        apply_haze_augmentation=False,
        num_synthetic_fallback=12,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PINNDepthNet(
        backbone=backbone,
        pretrained=True,
        is_physics_informed=is_physics_informed,
    ).to(device)

    criterion = PINNCombinedLoss(
        w_rank=1.0,
        w_phys=0.5 if is_physics_informed else 0.0,
        w_ssi=1.0,
        w_smooth=0.1,
        is_physics_informed=is_physics_informed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_total": [],
        "val_total": [],
        "train_ssi": [],
        "val_ssi": [],
        "train_rank": [],
        "val_rank": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_physics=is_physics_informed
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        history["train_ssi"].append(train_metrics.get("ssi", 0.0))
        history["val_ssi"].append(val_metrics.get("ssi", 0.0))
        history["train_rank"].append(train_metrics.get("rank", 0.0))
        history["val_rank"].append(val_metrics.get("rank", 0.0))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
            f"Train Total: {train_metrics['total']:.4f} | "
            f"Val Total: {val_metrics['total']:.4f} | "
            f"Val SSI: {val_metrics.get('ssi', 0.0):.4f} | "
            f"Val Rank: {val_metrics.get('rank', 0.0):.4f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "backbone": backbone,
                    "is_physics_informed": is_physics_informed,
                },
                checkpoint_path,
            )
            print(f"  -> Saved new best checkpoint to {checkpoint_path}")

    # Save training history JSON
    history_file = os.path.join(save_dir, f"history_{'pinn' if is_physics_informed else 'baseline'}.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete! Best checkpoint: {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physics-Informed Relative Depth CNN")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--baseline", action="store_true", help="Train CNN baseline without physics components")
    parser.add_argument("--dataset-dir", type=str, default="data", help="Path to KITTI/outdoor dataset folder")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone=args.backbone,
        is_physics_informed=not args.baseline,
        dataset_dir=args.dataset_dir,
    )

