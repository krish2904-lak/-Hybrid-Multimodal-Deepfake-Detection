"""
Unit tests for Federated Learning Module:
- FedAvg aggregation algorithm
- Federated client local training
- Server coordination
"""

import torch
import pytest
from src.federated.fedavg import aggregate_weights_fedavg
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.dataset import SyntheticDeepfakeDataset
from src.models.fusion_model import HybridMultimodalFusionModel


def test_fedavg_aggregation():
    # Two client parameter dictionaries with known values
    weights_c1 = {
        "layer1.weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "layer1.bias": torch.tensor([2.0, 2.0])
    }
    weights_c2 = {
        "layer1.weight": torch.tensor([[3.0, 3.0], [3.0, 3.0]]),
        "layer1.bias": torch.tensor([6.0, 6.0])
    }

    # Equal sample sizes (50-50): average should be exactly 2.0 and 4.0
    agg = aggregate_weights_fedavg([weights_c1, weights_c2], [10, 10])
    assert torch.allclose(agg["layer1.weight"], torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
    assert torch.allclose(agg["layer1.bias"], torch.tensor([4.0, 4.0]))

    # Unequal sample sizes (1:3): 0.25 * 1.0 + 0.75 * 3.0 = 2.5
    agg_weighted = aggregate_weights_fedavg([weights_c1, weights_c2], [10, 30])
    assert torch.allclose(agg_weighted["layer1.weight"], torch.tensor([[2.5, 2.5], [2.5, 2.5]]))


def test_federated_client_local_training():
    dataset = SyntheticDeepfakeDataset(num_samples=8, seq_length=4, image_size=112)
    client = FederatedClient(client_id="TestClient", dataset=dataset, batch_size=4)
    model = HybridMultimodalFusionModel(use_lightweight_video=True)

    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    res = client.train_round(initial_weights, model, local_epochs=1)

    assert res["client_id"] == "TestClient"
    assert res["num_samples"] == 8
    assert "weights" in res
    assert isinstance(res["local_loss"], float)
    assert isinstance(res["local_accuracy"], float)


def test_federated_server_single_round():
    server = FederatedServer(num_clients=2)
    server.setup_simulation(total_samples=16)
    results = server.run_rounds(num_rounds=1, local_epochs=1)

    assert len(results["rounds"]) == 1
    assert "global_accuracy" in results["rounds"][0]
    assert len(results["rounds"][0]["client_metrics"]) == 2
