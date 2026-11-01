"""
Adversarial Robustness Evaluation Benchmark
Evaluates model degradation and accuracy degradation curves under FGSM and PGD attacks.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from src.models.fusion_model import HybridMultimodalFusionModel
from src.models.dataset import SyntheticDeepfakeDataset
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.evaluation.metrics import calculate_metrics, compute_robustness_gap


def run_robustness_benchmark(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epsilons: List[float] = [0.0, 0.02, 0.05, 0.10],
    device: torch.device = torch.device("cpu"),
    output_dir: str = "outputs/adversarial"
) -> Dict[str, Any]:
    """
    Evaluates model accuracy across clean data and varying perturbation budgets.
    """
    model.eval()
    fgsm_attacker = FGSMAttack(model)
    pgd_attacker = PGDAttack(model)

    results: Dict[str, Any] = {
        "epsilons": epsilons,
        "clean_accuracy": 0.0,
        "fgsm_accuracies": [],
        "pgd_accuracies": [],
        "robustness_gaps_fgsm": [],
        "robustness_gaps_pgd": [],
        "sample_demonstration": {}
    }

    # Evaluate for each epsilon
    clean_acc_recorded = False

    for eps in epsilons:
        fgsm_correct = 0
        pgd_correct = 0
        clean_correct = 0
        total = 0

        first_sample_recorded = False

        for frames, audio, targets in data_loader:
            frames = frames.to(device)
            targets = targets.to(device)

            # 1. Clean forward
            with torch.no_grad():
                clean_out = model(frames, has_audio=False)
                clean_logits = clean_out["logits"]
                clean_preds = torch.argmax(clean_logits, dim=-1)
                clean_correct += (clean_preds == targets).sum().item()

            # 2. FGSM Attack
            if eps > 0:
                frames_fgsm = fgsm_attacker.attack_tensor(frames, targets, epsilon=eps)
            else:
                frames_fgsm = frames.clone()

            with torch.no_grad():
                fgsm_out = model(frames_fgsm, has_audio=False)
                fgsm_preds = torch.argmax(fgsm_out["logits"], dim=-1)
                fgsm_correct += (fgsm_preds == targets).sum().item()

            # 3. PGD Attack
            if eps > 0:
                frames_pgd = pgd_attacker.attack_tensor(frames, targets, epsilon=eps, num_steps=5)
            else:
                frames_pgd = frames.clone()

            with torch.no_grad():
                pgd_out = model(frames_pgd, has_audio=False)
                pgd_preds = torch.argmax(pgd_out["logits"], dim=-1)
                pgd_correct += (pgd_preds == targets).sum().item()

            total += targets.size(0)

            # Record a visual demonstration from first batch at eps=0.05
            if not first_sample_recorded and abs(eps - 0.05) < 1e-4:
                first_sample_recorded = True
                clean_prob = torch.softmax(clean_logits[0], dim=-1)[1].item() * 100.0
                fgsm_prob = torch.softmax(fgsm_out["logits"][0], dim=-1)[1].item() * 100.0
                pgd_prob = torch.softmax(pgd_out["logits"][0], dim=-1)[1].item() * 100.0

                results["sample_demonstration"] = {
                    "epsilon": eps,
                    "ground_truth": int(targets[0].item()),
                    "clean_prediction": int(clean_preds[0].item()),
                    "clean_fake_prob": round(clean_prob, 2),
                    "fgsm_prediction": int(fgsm_preds[0].item()),
                    "fgsm_fake_prob": round(fgsm_prob, 2),
                    "pgd_prediction": int(pgd_preds[0].item()),
                    "pgd_fake_prob": round(pgd_prob, 2),
                    "perturbation_l2_norm_fgsm": float(torch.norm(frames_fgsm[0] - frames[0]).item()),
                    "perturbation_l2_norm_pgd": float(torch.norm(frames_pgd[0] - frames[0]).item())
                }

        c_acc = clean_correct / max(1, total)
        f_acc = fgsm_correct / max(1, total)
        p_acc = pgd_correct / max(1, total)

        if not clean_acc_recorded:
            results["clean_accuracy"] = round(c_acc, 4)
            clean_acc_recorded = True

        results["fgsm_accuracies"].append(round(f_acc, 4))
        results["pgd_accuracies"].append(round(p_acc, 4))
        results["robustness_gaps_fgsm"].append(round(max(0.0, c_acc - f_acc), 4))
        results["robustness_gaps_pgd"].append(round(max(0.0, c_acc - p_acc), 4))

    # Save benchmark metrics to JSON
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "adversarial_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Adversarial Benchmark] Clean Acc: {results['clean_accuracy']:.4f} | "
          f"FGSM Accs: {results['fgsm_accuracies']} | PGD Accs: {results['pgd_accuracies']}")
    return results


if __name__ == "__main__":
    device = torch.device("cpu")
    model = HybridMultimodalFusionModel(
        video_feat_dim=256,
        audio_feat_dim=64,
        fusion_dim=128,
        num_classes=2,
        use_lightweight_video=True
    ).to(device)

    # Load trained checkpoint if exists
    ckpt_path = "checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")

    dataset = SyntheticDeepfakeDataset(num_samples=24, seq_length=4, image_size=112)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    run_robustness_benchmark(model, loader, device=device)
