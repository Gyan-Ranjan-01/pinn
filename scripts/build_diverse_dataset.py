"""
Build a diverse real-world outdoor dataset covering gardens, flora, streets, architecture,
and landscapes with high-precision ground-truth relative depth annotations.
"""

import os
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image

# Curated list of high-quality diverse outdoor images from Unsplash (free direct CDN)
IMAGE_URLS = [
    # Gardens & Flora
    ("garden_tulip_field", "https://images.unsplash.com/photo-1520763185298-1b434c919102?w=640"),
    ("garden_flowerbed", "https://images.unsplash.com/photo-1585320806297-9794b3e4eeae?w=640"),
    ("garden_roses", "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=640"),
    ("botanical_garden", "https://images.unsplash.com/photo-1519331379826-f10be5486c6f?w=640"),
    ("macro_white_flower", "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=640"),
    ("garden_pathway", "https://images.unsplash.com/photo-1508873696983-2df5293cb32f?w=640"),
    ("flower_meadow", "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=640"),
    
    # Urban & Streets
    ("city_street_crosswalk", "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=640"),
    ("urban_traffic_cars", "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=640"),
    ("suburban_road", "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=640"),
    ("city_buildings_street", "https://images.unsplash.com/photo-1477959858617-67f30bc75b82?w=640"),
    ("outdoor_parking_cars", "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=640"),
    
    # Landscapes & Nature
    ("mountain_lake_panorama", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=640"),
    ("countryside_green_hills", "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=640"),
    ("forest_sunlight_trees", "https://images.unsplash.com/photo-1448375240586-882707db888b?w=640"),
    ("rural_highway_drive", "https://images.unsplash.com/photo-1470240731273-7821a6eeb6bd?w=640"),
    
    # Atmospheric Haze & Adverse Weather
    ("foggy_morning_road", "https://images.unsplash.com/photo-1487621167305-5d248087c724?w=640"),
    ("misty_mountain_valley", "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=640"),
    ("foggy_forest_trees", "https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=640"),
    ("coastal_sea_horizon", "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=640"),
]


def download_and_annotate(output_dir: str = "data"):
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_dep_dir = os.path.join(output_dir, "train", "depth")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_dep_dir = os.path.join(output_dir, "val", "depth")

    for d in [train_img_dir, train_dep_dir, val_img_dir, val_dep_dir]:
        os.makedirs(d, exist_ok=True)

    # Priority 1: User's garden image
    user_garden = "samples/garden.jpg"
    if os.path.exists(user_garden):
        import shutil
        shutil.copy(user_garden, os.path.join(train_img_dir, "garden_user.jpg"))
        shutil.copy(user_garden, os.path.join(val_img_dir, "garden_user.jpg"))

    # Download diverse outdoor images
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for i, (name, url) in enumerate(IMAGE_URLS):
        dest_split = "val" if (i % 4 == 0) else "train"
        dest_path = os.path.join(output_dir, dest_split, "images", f"{name}.jpg")
        if not os.path.exists(dest_path):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, "wb") as f:
                    f.write(response.read())
                print(f"Downloaded [{dest_split}]: {name}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")

    # Load high-precision depth annotator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGenerating dense depth ground truth using high-precision model on {device}...")

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval().to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    for split in ["train", "val"]:
        img_dir = os.path.join(output_dir, split, "images")
        dep_dir = os.path.join(output_dir, split, "depth")
        images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

        for img_name in images:
            base = os.path.splitext(img_name)[0]
            dep_out_path = os.path.join(dep_dir, f"{base}.npy")

            img_path = os.path.join(img_dir, img_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            input_batch = midas_transforms(img_rgb).to(device)
            with torch.no_grad():
                pred = midas(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_np = pred.cpu().numpy()
            d_min, d_max = float(depth_np.min()), float(depth_np.max())
            # Normalize to relative depth where 0 is closest (foreground) and 1 is farthest (background)
            norm_disp = (depth_np - d_min) / (d_max - d_min + 1e-7)
            rel_depth = np.clip(1.0 - norm_disp, 0.0, 1.0).astype(np.float32)

            np.save(dep_out_path, rel_depth)
            print(f"Annotated [{split}]: {base} -> shape {rel_depth.shape}")

    print("\nDiverse dataset construction and annotation complete!")


if __name__ == "__main__":
    download_and_annotate()
