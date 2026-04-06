from __future__ import annotations

import torch
import torch.nn as nn


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 - dice_score_from_logits(logits, targets, eps=eps)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Focal Tversky Loss (Abraham & Khan, 2019)
#
# Addresses class imbalance in segmentation (brain = 95%+ of pixels,
# tumor = tiny fraction) by giving different weights to false positives
# and false negatives via the Tversky Index.
#
# Tversky Index = TP / (TP + alpha*FP + beta*FN)
#   - alpha > beta: penalize false positives more (conservative predictions)
#   - alpha < beta: penalize false negatives more (aggressive, catch all tumor)
#   - alpha = beta = 0.5: equivalent to standard Dice coefficient
#
# The "focal" exponent gamma < 1 makes the loss focus harder on difficult
# samples (slices where the model struggles), similar to Focal Loss for
# classification but applied to the Tversky index.
#
# Default: alpha=0.7, beta=0.3 — we penalize missing tumor (FN) more
# than over-predicting (FP), because missing a tumor is clinically worse.
# ---------------------------------------------------------------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 0.75, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha    # FP penalty weight
        self.beta = beta      # FN penalty weight (higher = catch more tumor)
        self.gamma = gamma    # focal exponent (< 1 focuses on hard examples)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        # Confusion matrix components per sample
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        # Tversky Index: generalized Dice with asymmetric FP/FN weighting
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

        # Focal modulation: (1 - TI)^gamma, focuses gradient on hard samples
        focal_tversky = (1.0 - tversky).pow(self.gamma)
        return focal_tversky.mean()

