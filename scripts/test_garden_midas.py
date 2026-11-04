import torch
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading MiDaS on:", device)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval().to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

img_bgr = cv2.imread("samples/garden.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

input_batch = midas_transforms(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_np = prediction.cpu().numpy()
print("Success! Depth shape:", depth_np.shape, "Min:", depth_np.min(), "Max:", depth_np.max())

# In MiDaS, higher values = closer (disparity). Let's normalize
d_min, d_max = depth_np.min(), depth_np.max()
norm_depth = (depth_np - d_min) / (d_max - d_min + 1e-8)

# Invert so 0 is close, 1 is far (standard relative depth convention)
rel_depth = 1.0 - norm_depth

# Save colorized preview using Turbo
colored = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imwrite("samples/garden_test_output.png", colored)
print("Saved preview to samples/garden_test_output.png")

