r"""
Fast Gradient Sign Method (FGSM) Adversarial Attack Module
Generates one-step gradient-based perturbations to evaluate model vulnerability:
x_{adv} = clamp(x + \epsilon * sign(\nabla_x L(x, y)), 0, 1)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class FGSMAttack:
    """
    FGSM (Fast Gradient Sign Method) Attack Generator.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.05):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def attack_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates adversarial perturbation for tensor x with ground truth y.
        x: (batch_size, C, H, W) or (batch_size, seq_len, C, H, W)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        if eps == 0:
            return x.clone().detach()

        x_adv = x.clone().detach().requires_grad_(True)

        self.model.eval()
        if x_adv.ndim == 5:
            # Video sequence: (B, seq_len, C, H, W)
            out = self.model(x_adv, has_audio=False) if hasattr(self.model, "video_branch") else self.model(x_adv)
            logits = out["logits"] if isinstance(out, dict) else (out[1] if isinstance(out, tuple) else out)
        else:
            # Single frame or batch of images: (B, C, H, W)
            if hasattr(self.model, "video_branch"):
                feats, logits = self.model.video_branch.cnn(x_adv)
            elif hasattr(self.model, "cnn"):
                feats, logits = self.model.cnn(x_adv)
            else:
                out = self.model(x_adv)
                logits = out["logits"] if isinstance(out, dict) else (out[1] if isinstance(out, tuple) else out)

        loss = self.loss_fn(logits, y)
        self.model.zero_grad()
        loss.backward()

        if x_adv.grad is None:
            return x.clone().detach()

        gradient_sign = x_adv.grad.data.sign()
        perturbed = x_adv + eps * gradient_sign
        return torch.clamp(perturbed, 0.0, 1.0).detach()
