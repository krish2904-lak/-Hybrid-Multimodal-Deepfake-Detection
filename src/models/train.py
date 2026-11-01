"""
Model Training and Validation Pipeline for Deepfake Detection
Includes checkpointing, early stopping, loss/accuracy tracking, and metric logging.
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple

from src.models.fusion_model import HybridMultimodalFusionModel
from src.models.dataset import SyntheticDeepfakeDataset, FolderDeepfakeDataset
from src.evaluation.metrics import calculate_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for frames, audio, targets in loader:
        frames = frames.to(device)
        audio = audio.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(frames, audio, has_audio=True)
        logits = output["logits"]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Runs a validation epoch and calculates full metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for frames, audio, targets in loader:
            frames = frames.to(device)
            audio = audio.to(device)
            targets = targets.to(device)

            output = model(frames, audio, has_audio=True)
            logits = output["logits"]
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
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs
    )
    metrics["loss"] = float(avg_loss)
    return metrics


def run_training(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    patience: int = 3,
    use_synthetic: bool = True
) -> Dict[str, Any]:
    """
    Executes complete training loop with early stopping and history logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Training] Initializing pipeline on device: {device}")

    # Dataset setup
    if use_synthetic:
        train_ds = SyntheticDeepfakeDataset(num_samples=32, seq_length=4, image_size=112)
        val_ds = SyntheticDeepfakeDataset(num_samples=16, seq_length=4, image_size=112)
    else:
        train_ds = FolderDeepfakeDataset("datasets/train", seq_length=4)
        val_ds = FolderDeepfakeDataset("datasets/validation", seq_length=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = HybridMultimodalFusionModel(
        video_feat_dim=256,
        audio_feat_dim=64,
        fusion_dim=128,
        num_classes=2,
        use_lightweight_video=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epochs": []
    }

    best_val_acc = 0.0
    patience_counter = 0

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pt")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Checkpoint saving
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_metrics": val_metrics
        }, latest_checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_metrics": val_metrics
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Training] Early stopping triggered at epoch {epoch}.")
                break

    # Save training history and latest metrics for dashboard
    history_file = os.path.join(output_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    val_metrics_file = os.path.join(output_dir, "validation_metrics.json")
    with open(val_metrics_file, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"[Training] Completed. Best Val Accuracy: {best_val_acc:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Multimodal Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
