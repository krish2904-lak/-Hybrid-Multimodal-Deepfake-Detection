r"""
Projected Gradient Descent (PGD) Adversarial Attack Module
Iterative gradient-based L-infinity bounded adversarial perturbation:
x^{t+1} = \Pi_{x + S} (x^t + \alpha * sign(\nabla_{x^t} L(x^t, y)))
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class PGDAttack:
    """
    PGD (Projected Gradient Descent) Attack Generator.
    Standard iterative white-box adversarial stress test.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.06,
        alpha: float = 0.015,
        num_steps: int = 10,
        random_start: bool = True
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def attack_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: Optional[float] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Executes multi-step PGD attack bounded by epsilon.
        x: (batch_size, C, H, W) or (batch_size, seq_len, C, H, W)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        steps = num_steps if num_steps is not None else self.num_steps

        if eps == 0:
            return x.clone().detach()

        x_orig = x.clone().detach()

        # Random start within epsilon ball
        if self.random_start:
            noise = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(x.device)
            x_adv = torch.clamp(x_orig + noise, 0.0, 1.0)
        else:
            x_adv = x_orig.clone()

        self.model.eval()

        for _ in range(steps):
            x_adv.requires_grad_(True)

            if x_adv.ndim == 5:
                out = self.model(x_adv, has_audio=False) if hasattr(self.model, "video_branch") else self.model(x_adv)
                logits = out["logits"] if isinstance(out, dict) else (out[1] if isinstance(out, tuple) else out)
            else:
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
                break

            # Step in direction of gradient
            grad_sign = x_adv.grad.data.sign()
            step_adv = x_adv.detach() + self.alpha * grad_sign

            # Project back onto L_infinity epsilon ball around original x
            eta = torch.clamp(step_adv - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + eta, min=0.0, max=1.0)

        return x_adv.detach()
