"""
Generates quantitative comparison charts and visual comparisons between
the CNN Baseline and the Physics-Informed CNN.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from pinn_depth.inference import DepthInferenceEngine, colorize_depth_map
from pinn_depth.physics.scattering import synthesize_hazy_image


def generate_results(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Training History Comparison
    hist_pinn_path = os.path.join("checkpoints", "history_pinn.json")
    hist_base_path = os.path.join("checkpoints", "history_baseline.json")

    if os.path.exists(hist_pinn_path) and os.path.exists(hist_base_path):
        with open(hist_pinn_path) as f:
            hp = json.load(f)
        with open(hist_base_path) as f:
            hb = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

        ep_pinn = range(1, len(hp["val_total"]) + 1)
        ep_base = range(1, len(hb["val_total"]) + 1)

        # Total Validation Loss
        axes[0].plot(ep_base, hb["val_total"], "o--", color="#64748b", label="Baseline CNN")
        axes[0].plot(ep_pinn, hp["val_total"], "s-", color="#2563eb", label="Physics-Informed CNN (Ours)")
        axes[0].set_title("Validation Total Loss vs Epochs", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Scale-and-Shift Invariant Loss (SSI)
        axes[1].plot(ep_base, hb["val_ssi"], "o--", color="#64748b", label="Baseline CNN")
        axes[1].plot(ep_pinn, hp["val_ssi"], "s-", color="#059669", label="Physics-Informed CNN (Ours)")
        axes[1].set_title("Scale-and-Shift Invariant Loss (SSI) vs Epochs", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("SSI Loss")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "training_comparison_curves.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved training curve comparison to: {chart_path}")

    # 2. Visual Comparison on Test Scenes
    eng_pinn = DepthInferenceEngine(
        checkpoint_path="checkpoints/best_pinn_model.pt",
        is_physics_informed=True,
    )
    eng_base = DepthInferenceEngine(
        checkpoint_path="checkpoints/best_baseline_model.pt",
        is_physics_informed=False,
    )

    test_samples = ["urban_street.jpg", "foggy_morning.jpg"]
    fig, axes = plt.subplots(len(test_samples), 4, figsize=(16, 4 * len(test_samples)), dpi=150)
    if len(test_samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, name in enumerate(test_samples):
        img_path = os.path.join("samples", name)
        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")

        # Predictions
        res_pinn = eng_pinn.predict(pil_img, colormap="turbo")
        res_base = eng_base.predict(pil_img, colormap="turbo")

        # Original RGB
        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(f"Input: {name.replace('_', ' ').title()}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Baseline CNN Depth
        axes[i, 1].imshow(res_base["colorized_depth"])
        axes[i, 1].set_title("CNN Baseline Depth", fontsize=10)
        axes[i, 1].axis("off")

        # PINN Depth
        axes[i, 2].imshow(res_pinn["colorized_depth"])
        axes[i, 2].set_title("Physics-Informed Depth (Ours)", fontsize=10, fontweight="bold", color="#1d4ed8")
        axes[i, 2].axis("off")

        # Physics Transmission Map
        if "transmission_map" in res_pinn:
            axes[i, 3].imshow(res_pinn["transmission_map"], cmap="gray", vmin=0, vmax=1)
            axes[i, 3].set_title(f"Physical Transmission (beta={res_pinn['scattering_beta']:.2f})", fontsize=10)
        else:
            axes[i, 3].imshow(res_pinn["depth_map"], cmap="gray")
            axes[i, 3].set_title("Raw Relative Depth Map", fontsize=10)
        axes[i, 3].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(output_dir, "visual_comparison.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"Saved visual comparison figure to: {vis_path}")


if __name__ == "__main__":
    generate_results()

