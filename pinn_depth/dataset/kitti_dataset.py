"""
Outdoor Depth Dataset Loader supporting KITTI and Procedural Benchmarks.
"""

from typing import Tuple, Optional, Callable, List
import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .synthetic_haze import SyntheticHazeAugmentation
from .sample_generator import generate_synthetic_outdoor_scene


class OutdoorDepthDataset(Dataset):
    """
    Dataset loader for outdoor scenes with RGB images and relative/metric depth maps.
    Supports KITTI format as well as synthetic benchmarks.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        target_size: Tuple[int, int] = (256, 384),
        apply_haze_augmentation: bool = True,
        haze_prob: float = 0.5,
        num_synthetic_fallback: int = 50,
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.apply_haze_aug = apply_haze_augmentation
        self.haze_aug = SyntheticHazeAugmentation(prob=haze_prob)
        self.num_synthetic_fallback = num_synthetic_fallback

        self.samples: List[Tuple[str, str]] = []

        if root_dir and os.path.exists(root_dir):
            split_dir = os.path.join(root_dir, split)
            target_dir = split_dir if os.path.exists(split_dir) else root_dir
            img_dir = os.path.join(target_dir, "images")
            dep_dir = os.path.join(target_dir, "depth")
            if os.path.exists(img_dir) and os.path.exists(dep_dir):
                img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
                for img_path in img_files:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    # Check for .npy or .png in depth directory
                    dep_npy = os.path.join(dep_dir, f"{base}.npy")
                    dep_png = os.path.join(dep_dir, f"{base}.png")
                    if os.path.exists(dep_npy):
                        self.samples.append((img_path, dep_npy))
                    elif os.path.exists(dep_png):
                        self.samples.append((img_path, dep_png))

        self.use_procedural = len(self.samples) == 0

    def __len__(self) -> int:
        if self.use_procedural:
            return self.num_synthetic_fallback
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        H, W = self.target_size

        if self.use_procedural:
            seed = (idx * 31 + 42) if self.split == "train" else (idx * 17 + 9999)
            img_np, dep_np = generate_synthetic_outdoor_scene(width=W, height=H, seed=seed)
        else:
            img_path, dep_path = self.samples[idx]
            img_bgr = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img_np, (W, H), interpolation=cv2.INTER_LINEAR)

            if dep_path.endswith(".npy"):
                dep_np = np.load(dep_path)
            else:
                dep_raw = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
                # Normalize 16-bit KITTI depth (depth in meters * 256)
                dep_np = dep_raw.astype(np.float32) / 256.0
                # Normalize to relative depth in [0, 1]
                max_d = np.percentile(dep_np[dep_np > 0], 95) if (dep_np > 0).any() else 80.0
                dep_np = np.clip(dep_np / max(1.0, max_d), 0.0, 1.0)

            dep_np = cv2.resize(dep_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # Convert to float tensors
        image_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
        depth_tensor = torch.from_numpy(dep_np).unsqueeze(0).float()

        # Data augmentation for training
        if self.split == "train":
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                depth_tensor = torch.flip(depth_tensor, dims=[-1])

        # Physics-informed haze augmentation
        if self.apply_haze_aug and self.split == "train":
            hazy_tensor, trans_tensor, beta, airlight = self.haze_aug(image_tensor, depth_tensor)
        else:
            hazy_tensor = image_tensor
            trans_tensor = torch.exp(-1.0 * depth_tensor)
            beta = 1.0
            airlight = torch.tensor([0.95, 0.95, 0.95]).view(3, 1, 1)

        valid_mask = (depth_tensor > 0.0) & torch.isfinite(depth_tensor)

        return {
            "image": hazy_tensor,
            "clear_image": image_tensor,
            "depth": depth_tensor,
            "transmission": trans_tensor,
            "beta": beta,
            "airlight": airlight,
            "valid_mask": valid_mask,
        }

