"""
Project Master Entrypoint & CLI Orchestrator
Hybrid Multimodal Federated Learning Framework for Robust Deepfake Detection under Adversarial Conditions

Usage:
  python main.py                    # Launches the interactive Streamlit Web Dashboard
  python main.py --predict <video>  # Runs deepfake detection on a target video
  python main.py --train            # Runs local model training loop
  python main.py --federated        # Runs simulated Federated Learning rounds (FedAvg)
  python main.py --adversarial      # Runs adversarial benchmark (FGSM & PGD)
"""

import sys
import os
import argparse
import subprocess


def run_dashboard():
    """Launches the Streamlit forensic dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    print(f"Launching Deepfake Forensic Dashboard via: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Deepfake Forensic Framework Master CLI")
    parser.add_argument("--web", action="store_true", help="Launch the Streamlit web dashboard (default)")
    parser.add_argument("--predict", type=str, help="Run deepfake inference on specified video path")
    parser.add_argument("--train", action="store_true", help="Run local model training pipeline")
    parser.add_argument("--federated", action="store_true", help="Run simulated federated learning rounds")
    parser.add_argument("--adversarial", action="store_true", help="Run FGSM & PGD robustness benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for training (default: 3)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds for federated simulation (default: 3)")

    args = parser.parse_args()

    # Ensure src is on sys.path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    if args.predict:
        from src.inference.predictor import DeepfakePredictor
        print(f"Analyzing video: {args.predict}")
        predictor = DeepfakePredictor()
        result = predictor.predict_video(args.predict)
        print("\n--- FORENSIC ANALYSIS RESULT ---")
        for k, v in result.items():
            if k not in ["face_crops_preview", "raw_frames_preview", "frame_results"]:
                print(f"  {k}: {v}")
        return

    if args.train:
        from src.models.train import run_training
        print("Starting model training pipeline...")
        run_training(epochs=args.epochs)
        return

    if args.federated:
        from src.federated.server import run_federated_simulation
        print("Starting federated learning simulation...")
        run_federated_simulation(num_rounds=args.rounds)
        return

    if args.adversarial:
        from src.attacks.robustness_bench import run_robustness_benchmark
        import torch
        from src.models.fusion_model import HybridMultimodalFusionModel
        from src.models.dataset import SyntheticDeepfakeDataset
        from torch.utils.data import DataLoader

        print("Starting adversarial robustness benchmark...")
        model = HybridMultimodalFusionModel(use_lightweight_video=True)
        dataset = SyntheticDeepfakeDataset(num_samples=16, seq_length=4, image_size=112)
        loader = DataLoader(dataset, batch_size=4)
        run_robustness_benchmark(model, loader)
        return

    # Default action: launch web dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
