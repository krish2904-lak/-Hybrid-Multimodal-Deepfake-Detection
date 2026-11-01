"""
Federated Learning Client Node Module
Represents an independent participating institution/device in the federated network.
Trains locally on private data partition without transferring raw media samples.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Tuple, Optional


class FederatedClient:
    """
    Simulated Federated Client holding private local dataset partition.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        batch_size: int = 4,
        device: Optional[torch.device] = None,
        lr: float = 1e-4
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.lr = lr
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train_round(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        model_blueprint: nn.Module,
        local_epochs: int = 1
    ) -> Dict[str, Any]:
        """
        Receives global weights, performs local training epochs, and returns
        updated model weights along with training diagnostics.
        """
        local_model = copy.deepcopy(model_blueprint).to(self.device)
        local_model.load_state_dict(global_state_dict)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(local_epochs):
            for frames, audio, targets in self.data_loader:
                frames = frames.to(self.device)
                audio = audio.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                output = local_model(frames, audio, has_audio=True)
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

        # Extract updated state dict to send back to server
        updated_weights = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}

        return {
            "client_id": self.client_id,
            "weights": updated_weights,
            "num_samples": self.num_samples,
            "local_loss": float(avg_loss),
            "local_accuracy": float(acc)
        }
