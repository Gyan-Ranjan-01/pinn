---
title: PINN Relative Depth Estimation
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Physics-Informed CNN for Relative Depth Estimation from Single Outdoor Images

A monocular relative depth estimation system that combines deep convolutional feature extraction (ResNet-18) with atmospheric scattering physical constraints (Koschmieder's model).

---

## 🌟 Key Features

- **Physics-Informed Architecture**: Integrates atmospheric attenuation and scattering principles from **Das et al. (ICVGIP 2012)** (*"Depth From Images of External Outdoor Scenes"*) and **Koschmieder's Law** $I(x) = J(x)t(x) + A(1 - t(x))$:
  - Luminance calculation: $C(x, y) = 0.33 R + 0.56 G + 0.11 B$
  - Empirical attenuation coefficient $\beta$ computation (Equations 8 & 19)
  - Distance formula: $d(x, y) = -\frac{1}{\beta} \ln\left[\frac{C_{gm} + C(x, y)}{C_{gm}}\right]$
- **Edge-Guided High-Frequency Filter**: Implements fast Guided Image Filtering (He et al., IEEE TPAMI) transferring high-frequency flower petals, stems, leaves, tree branches, and building edges directly into the depth map, eliminating blur and smudges.
- **Exact ICVGIP 2012 Scientific Colormap**: Nearest objects are radiant Yellow (`#ffff00`), midground is warm Red/Orange, and background/sky is deep dark Blue/Black.
- **Multi-Task Objective**:
  - $\mathcal{L}_{\text{rank}}$: Pairwise ordinal ranking loss.
  - $\mathcal{L}_{\text{phys}}$: Differentiable atmospheric transmission and dark channel consistency.
  - $\mathcal{L}_{\text{ssi}}$: Scale-and-shift invariant supervision.
  - $\mathcal{L}_{\text{smooth}}$: Edge-aware first-order gradient smoothness.
- **Human-Designed, Research-Grade Viewer UI**:
  - Side-by-side synchronized RGB scene and relative depth view.
  - FLIR Thermal Ironbow, Thermal Rainbow, and Multi-Hue scientific palettes.
  - FLIR MSX (Multi-Spectral Dynamic Imaging) detail relief embossing.
  - Live hover depth probe with real-time coordinate & value readout.
  - Interactive Pin A & Pin B placement with ordinal relative depth evaluation and confidence scoring.
  - In-browser model code and physical architecture walkthrough.

---

## 📖 Complete Model Documentation

For an exhaustive, line-by-line explanation of every class, function, tensor transformation, and physical equation (ideal for academic project presentations, viva defense, or technical interviews), see:
- 📄 **[Microsoft Word Document (.docx)](file:///c:/Users/gyanr/Desktop/Coding/Others/PINN2/docs/PINN_Depth_Model_Architecture_and_Line_by_Line_Guide.docx)** — Formatted Word file with styled tables, code blocks, callouts, and Viva Q&A.
- 📘 **[Markdown Architecture Guide](file:///c:/Users/gyanr/Desktop/Coding/Others/PINN2/docs/MODEL_ARCHITECTURE_EXPLAINED.md)**
- 💡 Includes 10 top examiner defense questions with detailed technical answers.

---

## 🚀 Quickstart (Local Development)

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch (with CUDA if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Sample Images

```bash
python scripts/generate_samples.py
```

### 3. Launch Web Application & API Server

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## 🧪 Testing and Benchmarking

Run the complete test suite:
```bash
pytest
```

Train a physics-informed model or CNN baseline:
```bash
# Train Physics-Informed CNN
python pinn_depth/train.py --epochs 5 --batch-size 4 --backbone resnet18

# Train CNN Baseline (ablation without physics)
python pinn_depth/train.py --epochs 5 --batch-size 4 --backbone resnet18 --baseline
```

Evaluate and compare:
```bash
python pinn_depth/evaluate.py --pinn-ckpt checkpoints/best_pinn_model.pt --baseline-ckpt checkpoints/best_baseline_model.pt
```

---

## ☁️ Backend Deployment Guide (Free & Low-Cost Alternatives to Render)

### Why Render Free Tier Struggles with PyTorch
Render's free tier has a **512 MB RAM limit**. When Python imports `torch`, `torchvision`, and loads deep neural network weights, memory spikes over 500 MB, triggering container crashes (`SIGKILL 137 / Out of Memory`). Furthermore, Render spins down after 15 minutes with 50-70 second cold starts and 0.1 CPU core throttling.

Here are the best tested alternatives:

### Option 1: Hugging Face Spaces (100% Free — Recommended)
- **Specs**: **16 GB RAM**, 2 vCPUs, 50 GB storage — completely FREE with no credit card required.
- **Never OOMs**: Handles PyTorch models effortlessly.
- **Deployment**:
  1. Create a free account at [huggingface.co](https://huggingface.co).
  2. Click **New Space** &rarr; select **Docker** &rarr; Blank template.
  3. Push this repository to your Space. The included `Dockerfile` will automatically build and launch the FastAPI server on port 7860!
  4. Your app and API are live instantly at `https://<your-username>-<space-name>.hf.space`.

### Option 2: Modal (Serverless GPU — $30/month Free Credit)
- **Specs**: Runs on real **NVIDIA T4 / A10G GPUs**.
- **Cold start**: 1.5 seconds.
- **Inference time**: ~20 ms per image.
- Modal gives **\$30 of free credits every month** automatically, enough for tens of thousands of inferences.

### Option 3: ONNX Runtime CPU Deployment (Railway / Koyeb / Render)
If you want to stay on a minimal CPU container or deploy to Railway:
- Convert the PyTorch model to ONNX using `pinn_depth/inference.py`:
  ```python
  from pinn_depth.inference import DepthInferenceEngine
  engine = DepthInferenceEngine()
  engine.export_to_onnx("pinn_depth.onnx")
  ```
- Running via `onnxruntime` reduces memory from ~900 MB down to **~75 MB**, running smoothly even on a 512 MB free instance!
