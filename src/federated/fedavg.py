"""
Federated Averaging (FedAvg) Aggregation Algorithm
Computes weighted parameter averaging across distributed client models:
W_{t+1} = \sum_{k=1}^K (n_k / N) * W_k
"""

import copy
import torch
from typing import List, Dict, Any


def aggregate_weights_fedavg(
    client_weights: List[Dict[str, torch.Tensor]],
    client_sample_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Computes FedAvg aggregation over a list of client state dicts.
    Parameters are weighted by each client's sample count.
    """
    if len(client_weights) == 0:
        raise ValueError("Cannot aggregate empty client weights list.")

    if len(client_weights) != len(client_sample_sizes):
        raise ValueError("client_weights and client_sample_sizes must have identical length.")

    total_samples = sum(client_sample_sizes)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be greater than 0.")

    # Deep copy the first client's weight structure as accumulator
    aggregated_weights = copy.deepcopy(client_weights[0])

    # Initialize all parameter tensors in accumulator to 0
    for key in aggregated_weights.keys():
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)

    # Accumulate weighted sums
    for weights, num_samples in zip(client_weights, client_sample_sizes):
        weight_factor = float(num_samples) / float(total_samples)
        for key in aggregated_weights.keys():
            aggregated_weights[key] += weights[key].to(torch.float32) * weight_factor

    # Convert back to original tensor dtypes
    ref_weights = client_weights[0]
    for key in aggregated_weights.keys():
        aggregated_weights[key] = aggregated_weights[key].to(ref_weights[key].dtype)

    return aggregated_weights
