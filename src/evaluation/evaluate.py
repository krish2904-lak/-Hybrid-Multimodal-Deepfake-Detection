"""
Model Evaluation and Benchmark Pipeline
Runs inference across test datasets and generates performance metrics.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.evaluation.metrics import calculate_metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    Evaluates a PyTorch model on a DataLoader.
    Supports both multimodal batches (frames, audio, labels) and frame-only batches.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    # frames, audio, labels
                    frames, audio, targets = batch
                    frames = frames.to(device)
                    audio = audio.to(device)
                    targets = targets.to(device)
                    output = model(frames, audio, has_audio=True)
                    logits = output["logits"] if isinstance(output, dict) else output
                elif len(batch) == 2:
                    # frames, targets
                    frames, targets = batch
                    frames = frames.to(device)
                    targets = targets.to(device)
                    output = model(frames)
                    logits = output["logits"] if isinstance(output, dict) else output
                else:
                    raise ValueError(f"Unexpected batch structure of length {len(batch)}")
            else:
                raise ValueError("Batch must be a list or tuple of tensors.")

            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n_samples = max(1, len(all_targets))
    avg_loss = total_loss / n_samples

    metrics = calculate_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs)
    )
    metrics["loss"] = float(avg_loss)

    return metrics


def save_evaluation_results(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Saves evaluation metrics dictionary to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
