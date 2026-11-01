"""
Unit tests for Adversarial Attacks (FGSM & PGD)
"""

import torch
import pytest

from src.models.cnn import LightweightFaceCNN
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack


def test_fgsm_attack_bounds():
    model = LightweightFaceCNN()
    attacker = FGSMAttack(model, epsilon=0.05)

    x = torch.rand(2, 3, 112, 112)
    y = torch.tensor([0, 1])

    x_adv = attacker.attack_tensor(x, y, epsilon=0.05)

    # Check that perturbation is strictly bounded by epsilon
    perturbation = torch.abs(x_adv - x)
    assert perturbation.max().item() <= 0.05 + 1e-5
    # Check that pixel values remain valid [0, 1]
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_pgd_attack_bounds():
    model = LightweightFaceCNN()
    attacker = PGDAttack(model, epsilon=0.08, alpha=0.02, num_steps=3)

    x = torch.rand(2, 3, 112, 112)
    y = torch.tensor([1, 0])

    x_adv = attacker.attack_tensor(x, y, epsilon=0.08, num_steps=3)

    perturbation = torch.abs(x_adv - x)
    assert perturbation.max().item() <= 0.08 + 1e-5
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0
