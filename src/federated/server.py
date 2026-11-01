"""
Federated Learning Central Server and Orchestrator
Coordinates communication rounds, client parameter aggregation via FedAvg,
and global convergence evaluation.
"""

import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Any, Optional

from src.models.fusion_model import HybridMultimodalFusionModel
from src.models.dataset import SyntheticDeepfakeDataset
from src.federated.client import FederatedClient
from src.federated.fedavg import aggregate_weights_fedavg
from src.evaluation.metrics import calculate_metrics


class FederatedServer:
    """
    Simulated Federated Learning Server managing 3+ independent nodes.
    """

    def __init__(
        self,
        num_clients: int = 3,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs"
    ):
        self.num_clients = num_clients
        self.device = device or torch.device("cpu")
        self.output_dir = output_dir

        # Initialize base model architecture
        self.global_model = HybridMultimodalFusionModel(
            video_feat_dim=256,
            audio_feat_dim=64,
            fusion_dim=128,
            num_classes=2,
            use_lightweight_video=True
        ).to(self.device)

        self.global_weights = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        self.clients: List[FederatedClient] = []
        self.test_loader: Optional[DataLoader] = None

    def setup_simulation(self, total_samples: int = 48) -> None:
        """
        Creates disjoint non-IID/balanced data partitions for simulated clients
        and a global evaluation test set.
        """
        full_dataset = SyntheticDeepfakeDataset(num_samples=total_samples, seq_length=4, image_size=112)
        test_dataset = SyntheticDeepfakeDataset(num_samples=16, seq_length=4, image_size=112)
        self.test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Partition indices across clients
        indices = np.arange(len(full_dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.num_clients)

        client_names = [
            "Client 1 (Forensic Lab Alpha)",
            "Client 2 (Media Platform Beta)",
            "Client 3 (Academic Node Gamma)"
        ]

        self.clients = []
        for i, split_idx in enumerate(splits):
            c_name = client_names[i] if i < len(client_names) else f"Client {i+1}"
            subset = Subset(full_dataset, split_idx.tolist())
            client = FederatedClient(
                client_id=c_name,
                dataset=subset,
                batch_size=4,
                device=self.device,
                lr=1e-4
            )
            self.clients.append(client)

    def evaluate_global_model(self) -> Dict[str, Any]:
        """Evaluates current global model on held-out test dataset."""
        if self.test_loader is None:
            raise RuntimeError("Simulation not setup. Call setup_simulation() first.")

        self.global_model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for frames, audio, targets in self.test_loader:
                frames = frames.to(self.device)
                audio = audio.to(self.device)
                targets = targets.to(self.device)

                output = self.global_model(frames, audio, has_audio=True)
                logits = output["logits"]
                loss = criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        metrics = calculate_metrics(
            y_true=all_targets,
            y_pred=all_preds,
            y_prob=all_probs
        )
        metrics["loss"] = float(total_loss / max(1, len(all_targets)))
        return metrics

    def run_rounds(self, num_rounds: int = 5, local_epochs: int = 1) -> Dict[str, Any]:
        """
        Executes communication rounds:
        Broadcast -> Local Train -> Collect -> FedAvg -> Global Evaluation
        """
        if len(self.clients) == 0:
            self.setup_simulation()

        round_history: List[Dict[str, Any]] = []

        print(f"[Federated Server] Starting {num_rounds} communication rounds with {len(self.clients)} clients...")

        for r in range(1, num_rounds + 1):
            client_updates = []
            client_weights = []
            client_sizes = []

            # Step 1 & 2: Distribute weights and train locally
            for client in self.clients:
                res = client.train_round(
                    global_state_dict=self.global_weights,
                    model_blueprint=self.global_model,
                    local_epochs=local_epochs
                )
                client_updates.append({
                    "client_id": res["client_id"],
                    "samples": res["num_samples"],
                    "loss": res["local_loss"],
                    "accuracy": res["local_accuracy"]
                })
                client_weights.append(res["weights"])
                client_sizes.append(res["num_samples"])

            # Step 3: FedAvg parameter aggregation
            self.global_weights = aggregate_weights_fedavg(client_weights, client_sizes)
            self.global_model.load_state_dict(self.global_weights)

            # Step 4: Evaluate global convergence
            global_eval = self.evaluate_global_model()

            round_entry = {
                "round": r,
                "global_accuracy": global_eval["accuracy"],
                "global_loss": global_eval["loss"],
                "global_f1": global_eval["f1"],
                "client_metrics": client_updates
            }
            round_history.append(round_entry)

            print(f"  Round {r:02d}/{num_rounds:02d} | "
                  f"Global Acc: {global_eval['accuracy']:.4f} | "
                  f"Global Loss: {global_eval['loss']:.4f}")

        # Save to outputs directory
        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, "federated_history.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"rounds": round_history, "num_clients": len(self.clients)}, f, indent=2)

        return {"rounds": round_history, "final_eval": global_eval}


def run_federated_simulation(num_rounds: int = 4) -> Dict[str, Any]:
    """Helper function to execute and return federated simulation results."""
    server = FederatedServer()
    server.setup_simulation()
    return server.run_rounds(num_rounds=num_rounds)


if __name__ == "__main__":
    run_federated_simulation(num_rounds=3)
