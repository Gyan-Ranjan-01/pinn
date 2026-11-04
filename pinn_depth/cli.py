"""
Command Line Interface for Physics-Informed Relative Depth Estimation.
"""

import sys
import argparse
import json
import cv2
import numpy as np
from PIL import Image

from pinn_depth.inference import DepthInferenceEngine, colorize_depth_map


def main():
    parser = argparse.ArgumentParser(
        prog="pinn-depth",
        description="Physics-Informed CNN for Relative Depth Estimation from Single Outdoor Images",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Estimate relative depth for an image")
    pred_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input outdoor RGB image")
    pred_parser.add_argument("--output", "-o", type=str, default="depth_prediction.png", help="Path to save output")
    pred_parser.add_argument("--colormap", "-c", type=str, default="thermal_ironbow", choices=["thermal_ironbow", "thermal_rainbow", "distinct", "das_icvgip", "turbo", "inferno", "viridis", "magma", "grayscale"])
    pred_parser.add_argument("--model", "-m", type=str, default="high_precision", choices=["high_precision", "custom_pinn"])
    pred_parser.add_argument("--gamma", type=float, default=1.35, help="Depth contrast gamma expansion factor")
    pred_parser.add_argument("--clahe", action="store_true", help="Apply adaptive histogram detail boost")
    pred_parser.add_argument("--contours", action="store_true", help="Overlay crisp object boundary contours")
    pred_parser.add_argument("--no-msx", action="store_true", help="Disable FLIR MSX detail embossing")
    pred_parser.add_argument("--msx-strength", type=float, default=0.55, help="FLIR MSX detail embossing intensity")
    pred_parser.add_argument("--ckpt", type=str, default="checkpoints/best_pinn_model.pt", help="Checkpoint path")

    # Command: compare
    comp_parser = subparsers.add_parser("compare", help="Compare relative depth between two image points")
    comp_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    comp_parser.add_argument("--p1", type=int, nargs=2, required=True, metavar=("X1", "Y1"), help="Point A (x y)")
    comp_parser.add_argument("--p2", type=int, nargs=2, required=True, metavar=("X2", "Y2"), help="Point B (x y)")
    comp_parser.add_argument("--model", "-m", type=str, default="high_precision", choices=["high_precision", "custom_pinn"])
    comp_parser.add_argument("--ckpt", type=str, default="checkpoints/best_pinn_model.pt")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI web application")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark comparing baseline vs PINN")
    bench_parser.add_argument("--pinn-ckpt", type=str, default="checkpoints/best_pinn_model.pt")
    bench_parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/best_baseline_model.pt")

    args = parser.parse_args()

    if args.command == "predict":
        engine = DepthInferenceEngine(checkpoint_path=args.ckpt, model_type=args.model)
        res = engine.predict(
            args.image,
            colormap=args.colormap,
            gamma=args.gamma,
            use_clahe=args.clahe,
            show_contours=args.contours,
        )
        cv2.imwrite(args.output, cv2.cvtColor(res["colorized_depth"], cv2.COLOR_RGB2BGR))
        print(f"Predicted depth map saved to: {args.output}")
        print(f"Model used: {res.get('model_used', args.model)}")
        print(f"Stats: Min={res['min_depth']:.3f}, Mean={res['mean_depth']:.3f}, Max={res['max_depth']:.3f}")
        if "scattering_beta" in res:
            print(f"Atmospheric parameters: Airlight={res['airlight']}, beta={res['scattering_beta']:.2f}")

    elif args.command == "compare":
        engine = DepthInferenceEngine(checkpoint_path=args.ckpt, model_type=args.model)
        res = engine.predict(args.image)
        comp = engine.compare_points(res["depth_map"], args.p1, args.p2)
        print("\n=== Relative Depth Ordering Analysis ===")
        print(f"Point A ({args.p1[0]}, {args.p1[1]}): Relative Depth = {comp['point_a']['relative_depth']:.3f}")
        print(f"Point B ({args.p2[0]}, {args.p2[1]}): Relative Depth = {comp['point_b']['relative_depth']:.3f}")
        print(f"Delta: {comp['delta_depth']:.3f} | Confidence: {comp['confidence'] * 100:.1f}%")
        print(f"Outcome: {comp['relationship']}\n")

    elif args.command == "serve":
        import uvicorn
        from api.app import app
        print(f"Starting server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "benchmark":
        from pinn_depth.evaluate import compare_models
        compare_models(pinn_checkpoint=args.pinn_ckpt, baseline_checkpoint=args.baseline_ckpt)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

