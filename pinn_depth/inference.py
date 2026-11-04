"""
Inference Engine and Visualization Utilities for Relative Depth Estimation.
"""

from typing import Tuple, Dict, Any, Optional, List
import io
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
import torch

from .models.pinn_net import PINNDepthNet
from .physics.scattering import depth_to_transmission, recover_radiance
from .physics.dark_channel import estimate_atmospheric_light_torch
from .physics.das_physics import (
    compute_luminance_das,
    compute_das_attenuation_beta,
    fast_guided_filter,
    colorize_das_paper,
    colorize_distinct,
    colorize_flir_ironbow,
    colorize_flir_rainbow,
    apply_flir_msx_detail_embossing,
    enhance_depth_contrast,
    overlay_depth_boundaries,
)


COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
}


def colorize_depth_map(
    depth: np.ndarray,
    colormap_name: str = "thermal_ironbow",
    invert: bool = True,
) -> np.ndarray:
    """
    Colorizes a normalized relative depth map [0, 1] into an RGB visualization.

    Args:
        depth: (H, W) float array in [0, 1].
        colormap_name: 'thermal_ironbow' (FLIR Ironbow thermal camera palette),
                       'thermal_rainbow' (FLIR high-contrast thermal rainbow),
                       'distinct' (High-contrast multi-hue object separation),
                       'das_icvgip' (Das et al. ICVGIP 2012 yellow-red-black),
                       'turbo', 'inferno', 'viridis', 'magma', or 'grayscale'.
        invert: If True, closer objects (smaller depth) receive warm/bright colors,
                farther objects receive cool/dark colors.

    Returns:
        colorized: (H, W, 3) uint8 RGB image.
    """
    d_norm = np.clip(depth, 0.0, 1.0)

    # 1. FLIR Thermal Ironbow (White-Hot -> Pale Yellow -> Orange -> Crimson -> Purple -> Black)
    if colormap_name.lower() in ("thermal_ironbow", "ironbow", "thermal", "flir", "flir_ironbow"):
        d_input = d_norm if invert else (1.0 - d_norm)
        return colorize_flir_ironbow(d_input)

    # 2. FLIR Thermal Rainbow High-Contrast
    if colormap_name.lower() in ("thermal_rainbow", "flir_rainbow", "rainbow_hc"):
        d_input = d_norm if invert else (1.0 - d_norm)
        return colorize_flir_rainbow(d_input)

    # 3. High-Distinction Multi-Hue Colormap (Yellow -> Amber -> Coral -> Magenta -> Violet -> Azure -> Navy -> Black)
    if colormap_name.lower() in ("distinct", "multi_hue", "objects"):
        d_input = d_norm if invert else (1.0 - d_norm)
        return colorize_distinct(d_input)

    # 4. Das et al. (ICVGIP 2012) Exact Colormap (Yellow = Near, Dark Black = Far)
    if colormap_name.lower() in ("das_icvgip", "icvgip", "paper"):
        d_input = d_norm if invert else (1.0 - d_norm)
        return colorize_das_paper(d_input)

    if invert:
        # 1.0 is near (warm), 0.0 is far (cool)
        d_vis = 1.0 - d_norm
    else:
        d_vis = d_norm

    d_uint8 = (d_vis * 255.0).astype(np.uint8)

    if colormap_name.lower() == "grayscale":
        return np.repeat(d_uint8[:, :, np.newaxis], 3, axis=2)

    cv_cmap = COLORMAPS.get(colormap_name.lower(), cv2.COLORMAP_TURBO)
    colored_bgr = cv2.applyColorMap(d_uint8, cv_cmap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return colored_rgb


class DepthInferenceEngine:
    """
    High-performance inference engine for Relative Depth Estimation.
    Supports both High-Precision Vision-Transformer PINN model and custom ResNet-18 PINN.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "high_precision",
        backbone: str = "resnet18",
        is_physics_informed: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_type = model_type
        self.hp_model = None
        self.hp_transforms = None

        if model_type == "high_precision":
            try:
                # DPT-Hybrid (Dense Prediction Transformer + ResNet-50) for fine object separation
                self.hp_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True).eval().to(self.device)
                self.hp_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
            except Exception as e:
                print(f"Notice: Loading MiDaS_small fallback ({e}).")
                try:
                    self.hp_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval().to(self.device)
                    self.hp_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                except Exception as e2:
                    print(f"Warning: Falling back to custom PINN ({e2}).")
                    self.model_type = "custom_pinn"

        # Initialize custom PINN model
        self.model = PINNDepthNet(
            backbone=backbone,
            pretrained=(checkpoint_path is None or not os.path.exists(str(checkpoint_path))),
            is_physics_informed=is_physics_informed,
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)

        self.model.eval()

    def predict(
        self,
        image_input: Image.Image | np.ndarray | str,
        colormap: str = "thermal_ironbow",
        target_size: Optional[Tuple[int, int]] = None,
        model_type: Optional[str] = None,
        invert: bool = True,
        gamma: float = 1.35,
        use_clahe: bool = False,
        show_contours: bool = False,
        contour_threshold: float = 0.045,
        msx_detail: bool = True,
        msx_strength: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Runs relative depth inference with adaptive dynamic range tone expansion
        and FLIR MSX detail embossing for authentic thermal camera clarity.

        Args:
            image_input: PIL Image, RGB numpy array, or file path.
            colormap: Colormap ('thermal_ironbow', 'thermal_rainbow', 'distinct', etc.).
            target_size: Optional (H, W) processing resolution.
            model_type: Override default model ('high_precision' or 'custom_pinn').
            invert: If True, warm/white colors indicate near objects, dark/cool indicate far.
            gamma: Depth tone expansion factor (default 1.35 spreads foreground items).
            use_clahe: If True, applies local adaptive contrast enhancement.
            show_contours: If True, overlays depth-boundary outlines on distinct objects.
            contour_threshold: Gradient threshold for object depth edges.
            msx_detail: If True, applies FLIR MSX multi-spectral edge detail embossing.
            msx_strength: Edge relief intensity (default 0.45).

        Returns:
            Dictionary with relative depth, colorized depth, transmission, and physics metrics.
        """
        active_type = model_type or self.model_type

        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")

        orig_w, orig_h = orig_pil.size
        img_np_orig = np.array(orig_pil)

        # 1. High-Precision Transformer/CNN Physics-Guided Pipeline
        if active_type == "high_precision" and self.hp_model is not None:
            input_batch = self.hp_transforms(img_np_orig).to(self.device)

            with torch.no_grad():
                pred = self.hp_model(input_batch)
                # Interpolate back to original resolution with bicubic filtering
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_np = pred.cpu().numpy()
            # Robust percentile normalization (p1 to p99) prevents distant sky/horizon from compressing foreground
            p1, p99 = np.percentile(depth_np, 1), np.percentile(depth_np, 99)
            norm_disp = np.clip((depth_np - p1) / (p99 - p1 + 1e-7), 0.0, 1.0)
            depth_orig = np.clip(1.0 - norm_disp, 0.0, 1.0)

            # Atmospheric Physics Decomposition (Koschmieder Model)
            tensor_orig = torch.from_numpy(img_np_orig.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device) / 255.0
            airlight_vec = estimate_atmospheric_light_torch(tensor_orig).squeeze().cpu().numpy().tolist()
            beta_val = 1.25

        # 2. Custom ResNet-18 PINN CNN Pipeline
        else:
            proc_h, proc_w = target_size if target_size else (320, 512)
            resized_pil = orig_pil.resize((proc_w, proc_h), Image.Resampling.BILINEAR)

            img_np = np.array(resized_pil, dtype=np.float32) / 255.0
            tensor_in = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor_in)
                depth_tensor = output["depth"]
                depth_raw = depth_tensor.squeeze().cpu().numpy()

                depth_orig = cv2.resize(depth_raw, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                depth_orig = np.clip(depth_orig, 0.0, 1.0)

            if "airlight" in output:
                airlight_vec = output["airlight"].squeeze().cpu().numpy().tolist()
                beta_val = float(output["beta"].squeeze().cpu().numpy())
            else:
                airlight_vec = [0.95, 0.95, 0.95]
                beta_val = 1.0

        # Dynamic Range & Gamma Expansion (prevents foreground bunching, separates individual objects)
        depth_orig = enhance_depth_contrast(depth_orig, gamma=gamma, use_clahe=use_clahe)

        # High-frequency edge-preserving Guided Filter using original RGB guidance
        # Sharpens individual flower petals, stems, leaves, tree branches, and architecture contours
        guide_gray = cv2.cvtColor(img_np_orig, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth_orig = fast_guided_filter(guide_gray, depth_orig, radius=4, eps=1e-4)

        # Das et al. (ICVGIP 2012) empirical attenuation coefficient
        c_lum = compute_luminance_das(img_np_orig)
        das_beta, _ = compute_das_attenuation_beta(c_lum)

        # Physical Transmission Map: t(x) = exp(-beta * d(x))
        trans_np = np.clip(np.exp(-beta_val * depth_orig), 0.0, 1.0)

        colorized_rgb = colorize_depth_map(depth_orig, colormap_name=colormap, invert=invert)

        # FLIR MSX (Multi-Spectral Dynamic Imaging) detail embossing
        # Embosses visible edge relief onto the thermal depth map for true thermal camera clarity
        if msx_detail:
            colorized_rgb = apply_flir_msx_detail_embossing(
                colorized_rgb, img_np_orig, strength=msx_strength
            )

        # Optional object boundary edge overlay
        if show_contours:
            colorized_rgb = overlay_depth_boundaries(
                colorized_rgb, depth_orig, threshold=contour_threshold, outline_color=(255, 255, 255)
            )

        result: Dict[str, Any] = {
            "depth_map": depth_orig,
            "colorized_depth": colorized_rgb,
            "min_depth": float(np.min(depth_orig)),
            "max_depth": float(np.max(depth_orig)),
            "mean_depth": float(np.mean(depth_orig)),
            "image_size": [orig_w, orig_h],
            "airlight": airlight_vec,
            "scattering_beta": beta_val,
            "das_beta": das_beta,
            "transmission_map": trans_np,
            "model_used": active_type,
        }

        return result

    def compare_points(
        self,
        depth_map: np.ndarray,
        point_a: Tuple[int, int],
        point_b: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Determines the relative ordering between two image coordinates:
            point_a: (x_a, y_a)
            point_b: (x_b, y_b)

        Returns:
            Dictionary explaining which object is closer/farther and ordinal confidence.
        """
        H, W = depth_map.shape
        xa = max(0, min(W - 1, int(point_a[0])))
        ya = max(0, min(H - 1, int(point_a[1])))

        xb = max(0, min(W - 1, int(point_b[0])))
        yb = max(0, min(H - 1, int(point_b[1])))

        da = float(depth_map[ya, xa])
        db = float(depth_map[yb, xb])

        delta = abs(da - db)

        if da < db - 0.02:
            closer = "A"
            farther = "B"
            relationship = "Point A is significantly CLOSER to the camera than Point B."
        elif db < da - 0.02:
            closer = "B"
            farther = "A"
            relationship = "Point B is significantly CLOSER to the camera than Point A."
        else:
            closer = "Tie"
            farther = "Tie"
            relationship = "Point A and Point B are at approximately the same relative distance."

        confidence = min(1.0, delta / 0.15)

        return {
            "point_a": {"x": xa, "y": ya, "relative_depth": da},
            "point_b": {"x": xb, "y": yb, "relative_depth": db},
            "delta_depth": float(delta),
            "closer": closer,
            "farther": farther,
            "relationship": relationship,
            "confidence": float(confidence),
        }

    def export_to_onnx(
        self,
        output_path: str = "pinn_depth.onnx",
        input_size: Tuple[int, int] = (256, 384),
    ) -> str:
        """
        Exports the model to ONNX for lightweight CPU deployment (e.g. Hugging Face / Vercel / Railway).
        """
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=self.device)
        self.model.eval()

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={
                "image": {0: "batch_size", 2: "height", 3: "width"},
                "depth": {0: "batch_size", 2: "height", 3: "width"},
            },
            opset_version=14,
        )
        return output_path

