from __future__ import annotations

# losses.py — Segmentation loss functions
#
# Brain tumor segmentation has extreme class imbalance:
# tumor pixels are typically <5% of a full brain slice.
# Standard cross-entropy treats all pixels equally, so the model
# can get low loss by simply predicting "no tumor everywhere".
#
# Both loss functions below are designed to handle this imbalance:
# - DiceBCELoss:      focuses on region overlap (shape) + per-pixel accuracy
# - FocalTverskyLoss: aggressively penalizes missed tumor pixels (false negatives)
#
# Both accept raw logits (before sigmoid) for numerical stability.

import torch
import torch.nn as nn


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Dice coefficient from raw logits and binary targets.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    where A = predicted probabilities, B = binary ground truth mask.

    Using soft probabilities (not hard 0/1 predictions) makes this differentiable,
    so gradients can flow through it during backpropagation.

    Returns the mean Dice score across all samples in the batch.
    """
    probs = torch.sigmoid(logits)
    # Flatten spatial dims so we compute per-sample Dice: (B, C, H, W) -> (B, H*W)
    probs   = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)         # element-wise overlap
    union        = probs.sum(dim=1) + targets.sum(dim=1) # total predicted + actual tumor
    dice = (2.0 * intersection + eps) / (union + eps)   # eps avoids 0/0 on empty slices
    return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss = 1 - Dice score. Minimizing this maximizes overlap with the target."""
    return 1.0 - dice_score_from_logits(logits, targets, eps=eps)


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss.

    Loss = dice_weight * DiceLoss + bce_weight * BCELoss
    Default: equal weighting (0.5 + 0.5 = 1.0)

    Why combine them?
    - Dice loss focuses on region overlap — good for overall tumor shape,
      handles class imbalance well (cares about shape, not individual pixels)
    - BCE (binary cross-entropy) focuses on per-pixel classification accuracy —
      provides dense gradients for every pixel, which stabilizes early training

    Together they balance global shape accuracy with local pixel precision.
    BCEWithLogitsLoss applies sigmoid internally — numerically more stable than
    computing sigmoid first and then applying BCE separately.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.bce = nn.BCEWithLogitsLoss()  # numerically stable BCE with built-in sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, targets)
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
    """Focal Tversky Loss for highly imbalanced binary segmentation.

    Tversky Index (TI) = TP / (TP + alpha*FP + beta*FN)
    Focal Tversky Loss = mean( (1 - TI)^gamma )

    Args:
        alpha: weight for false positives (FP penalty). Default 0.7.
        beta:  weight for false negatives (FN penalty). Default 0.3.
               NOTE: alpha + beta should = 1.0 for stability.
               Higher beta = model is penalized more for missing tumor pixels.
        gamma: focal exponent. gamma < 1 increases focus on hard examples.
               gamma = 1 gives standard Tversky loss.
        eps:   small constant to prevent division by zero on empty masks.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 0.75, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha    # FP penalty weight — lower = more tolerant of false alarms
        self.beta  = beta     # FN penalty weight — higher = must catch all tumor pixels
        self.gamma = gamma    # focal exponent — lower = harder focus on difficult slices
        self.eps   = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W)
        probs   = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        # Compute confusion matrix components per sample in the batch
        tp = (probs * targets).sum(dim=1)              # true positives: predicted tumor, is tumor
        fp = (probs * (1.0 - targets)).sum(dim=1)      # false positives: predicted tumor, is background
        fn = ((1.0 - probs) * targets).sum(dim=1)      # false negatives: predicted background, is tumor

        # Tversky Index: generalized Dice with asymmetric FP/FN weighting
        # With alpha=0.7, beta=0.3: FN is penalized more than FP
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

        # Focal modulation: (1 - TI)^gamma
        # When TI is low (model struggling), (1-TI) is large -> amplified loss -> harder training signal
        # When TI is high (easy example), (1-TI) is small -> down-weighted -> model focuses elsewhere
        focal_tversky = (1.0 - tversky).pow(self.gamma)
        return focal_tversky.mean()
