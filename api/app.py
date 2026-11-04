"""
FastAPI Server for Physics-Informed Relative Depth Estimation.
"""

from typing import Optional, List, Dict, Any
import os
import io
import glob
from contextlib import asynccontextmanager
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from pinn_depth.inference import DepthInferenceEngine, colorize_depth_map
from api.utils import numpy_to_base64_png, depth_to_point_cloud_json

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload engine on server start
    get_engine()
    yield

app = FastAPI(
    title="Physics-Informed Relative Depth API",
    description="Monocular relative depth estimation using Physics-Informed CNN (Koschmieder atmospheric model)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
engine: Optional[DepthInferenceEngine] = None
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def get_engine() -> DepthInferenceEngine:
    global engine
    if engine is None:
        ckpt_path = os.path.join("checkpoints", "best_pinn_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        engine = DepthInferenceEngine(
            checkpoint_path=ckpt_path,
            backbone="resnet18",
            is_physics_informed=True,
        )
    return engine


class ComparePointsRequest(BaseModel):
    point_a: List[float]  # [x, y] in range [0, 1] relative or pixel coordinates
    point_b: List[float]
    depth_grid: List[List[float]]





@app.get("/health")
def health_check():
    eng = get_engine()
    return {
        "status": "healthy",
        "device": str(eng.device),
        "backbone": eng.model.backbone_name,
        "is_physics_informed": eng.model.is_physics_informed,
    }


@app.get("/api/samples")
def get_sample_images():
    """
    Returns list of preloaded outdoor sample images.
    """
    if not os.path.exists(SAMPLE_DIR):
        return []

    image_paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")) + glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
    samples = []
    for p in image_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        title = name.replace("_", " ").title()
        samples.append({
            "id": name,
            "title": title,
            "filename": os.path.basename(p),
            "url": f"/samples/{os.path.basename(p)}",
        })
    return samples


@app.post("/api/predict")
async def predict_depth(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    colormap: str = Form("thermal_ironbow"),
    invert: bool = Form(True),
    model_type: str = Form("high_precision"),
    gamma: float = Form(1.35),
    use_clahe: bool = Form(False),
    show_contours: bool = Form(False),
    msx_detail: bool = Form(True),
    msx_strength: float = Form(0.55),
):
    """
    Processes an image and produces a relative depth map and physical atmospheric parameters.
    """
    eng = get_engine()

    try:
        if file is not None:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        elif sample_id:
            sample_path = os.path.join(SAMPLE_DIR, f"{sample_id}.jpg")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(SAMPLE_DIR, f"{sample_id}.png")
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample image not found.")
            pil_image = Image.open(sample_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Must provide an image file or sample_id.")

        orig_w, orig_h = pil_image.size
        # Run inference
        result = eng.predict(
            pil_image,
            colormap=colormap,
            invert=invert,
            model_type=model_type,
            gamma=gamma,
            use_clahe=use_clahe,
            show_contours=show_contours,
            msx_detail=msx_detail,
            msx_strength=msx_strength,
        )

        depth_map = result["depth_map"]
        colorized_rgb = result["colorized_depth"]

        # Downsample depth grid to 64x64 for fast interactive hover probing on client
        probe_grid = cv2.resize(depth_map, (64, 48), interpolation=cv2.INTER_AREA).tolist()

        # Encode RGB images to Base64
        depth_b64 = numpy_to_base64_png(colorized_rgb)
        orig_b64 = numpy_to_base64_png(np.array(pil_image))

        # Transmission visualization
        trans_b64 = None
        if "transmission_map" in result:
            trans_vis = (result["transmission_map"] * 255.0).astype(np.uint8)
            trans_rgb = np.repeat(trans_vis[:, :, np.newaxis], 3, axis=2)
            trans_b64 = numpy_to_base64_png(trans_rgb)

        return {
            "status": "success",
            "image_size": [orig_w, orig_h],
            "original_image": orig_b64,
            "depth_map_png": depth_b64,
            "transmission_png": trans_b64,
            "probe_grid": probe_grid,
            "statistics": {
                "min": result["min_depth"],
                "max": result["max_depth"],
                "mean": result["mean_depth"],
            },
            "physics": {
                "airlight": result.get("airlight", [0.95, 0.95, 0.95]),
                "scattering_beta": result.get("scattering_beta", 1.0),
                "das_beta": result.get("das_beta", 0.45),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/compare")
def compare_two_points(req: ComparePointsRequest):
    """
    Computes relative ordinal ranking between two coordinates on the predicted depth map.
    """
    grid = np.array(req.depth_grid, dtype=np.float32)
    H, W = grid.shape

    # Handle coordinates normalized to [0, 1] or raw indices
    ax, ay = req.point_a
    bx, by = req.point_b

    xa = int(ax * (W - 1)) if ax <= 1.0 else min(W - 1, int(ax))
    ya = int(ay * (H - 1)) if ay <= 1.0 else min(H - 1, int(ay))

    xb = int(bx * (W - 1)) if bx <= 1.0 else min(W - 1, int(xb))
    yb = int(by * (H - 1)) if by <= 1.0 else min(H - 1, int(by))

    da = float(grid[ya, xa])
    db = float(grid[yb, xb])
    delta = abs(da - db)

    if da < db - 0.03:
        closer = "Point A (Closer)"
        farther = "Point B (Farther)"
        summary = f"Point A is closer to the camera than Point B (relative depth: {da:.2f} vs {db:.2f})."
    elif db < da - 0.03:
        closer = "Point B (Closer)"
        farther = "Point A (Farther)"
        summary = f"Point B is closer to the camera than Point A (relative depth: {db:.2f} vs {da:.2f})."
    else:
        closer = "Equivalent"
        farther = "Equivalent"
        summary = f"Point A and Point B are at approximately the same relative distance ({da:.2f} ≈ {db:.2f})."

    confidence = min(1.0, delta / 0.15)

    return {
        "point_a": {"coord": [xa, ya], "depth": da},
        "point_b": {"coord": [xb, yb], "depth": db},
        "delta": delta,
        "closer": closer,
        "farther": farther,
        "summary": summary,
        "confidence_score": round(confidence * 100, 1),
    }


@app.post("/api/mesh-3d")
async def get_3d_mesh(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    """
    Generates a 3D vertex heightfield for interactive WebGL / Three.js display.
    """
    eng = get_engine()
    if file is not None:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    elif sample_id:
        sample_path = os.path.join(SAMPLE_DIR, f"{sample_id}.jpg")
        pil_image = Image.open(sample_path).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Missing image")

    result = eng.predict(pil_image)
    depth = result["depth_map"]
    img_rgb = np.array(pil_image)

    cloud = depth_to_point_cloud_json(depth, img_rgb, max_resolution=100)
    return cloud


# Serve samples directory
if os.path.exists(SAMPLE_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLE_DIR), name="samples")

# Serve Web Frontend if directory exists
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
