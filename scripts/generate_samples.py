"""
Generate high quality outdoor sample images for the web app gallery.
"""

import os
import cv2
import numpy as np
import torch
from pinn_depth.dataset.sample_generator import generate_synthetic_outdoor_scene
from pinn_depth.physics.scattering import synthesize_hazy_image


def generate_samples(output_dir: str = "samples"):
    os.makedirs(output_dir, exist_ok=True)

    scenes = [
        {"name": "urban_street", "seed": 101, "cars": 3, "trees": 2, "fog": False},
        {"name": "highway_drive", "seed": 204, "cars": 4, "trees": 1, "fog": False},
        {"name": "country_road", "seed": 305, "cars": 1, "trees": 6, "fog": False},
        {"name": "foggy_morning", "seed": 408, "cars": 2, "trees": 3, "fog": True, "beta": 1.6},
    ]

    for sc in scenes:
        img_np, dep_np = generate_synthetic_outdoor_scene(
            width=512,
            height=320,
            num_cars=sc["cars"],
            num_trees=sc["trees"],
            seed=sc["seed"],
        )

        if sc.get("fog", False):
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            dep_t = torch.from_numpy(dep_np).unsqueeze(0).unsqueeze(0).float()
            airlight = torch.tensor([0.92, 0.94, 0.96]).view(1, 3, 1, 1)
            hazy_t, _ = synthesize_hazy_image(img_t, dep_t, airlight=airlight, beta=sc.get("beta", 1.5))
            img_np = (hazy_t.squeeze().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        out_img_path = os.path.join(output_dir, f"{sc['name']}.jpg")
        cv2.imwrite(out_img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f"Generated sample: {out_img_path}")


if __name__ == "__main__":
    generate_samples()

