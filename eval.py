from __future__ import annotations

# eval.py — Segmentation evaluation metrics
#
# Computes Dice coefficient, IoU (Jaccard index), Precision, and Recall
# over the entire validation set in a single pass (not averaged per-batch).
#
# Why accumulate counts instead of averaging per batch?
# Per-batch averaging gives different results depending on batch size and
# class distribution within each batch (small batches with no tumor get
# artificially high Dice). Accumulating TP/FP/FN globally gives the
# correct dataset-level metric regardless of batch size.

from typing import Dict

import torch


def _metrics_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-6) -> Dict[str, float]:
    """Compute all four metrics from accumulated confusion matrix counts.

    Formulas:
      Dice      = 2*TP / (2*TP + FP + FN)  — harmonic mean of precision and recall
      IoU       = TP / (TP + FP + FN)       — stricter than Dice; always lower
      Precision = TP / (TP + FP)            — of all predicted tumor pixels, how many are real?
      Recall    = TP / (TP + FN)            — of all real tumor pixels, how many did we catch?

    eps prevents division by zero when TP=FP=FN=0 (model predicts nothing).
    """
    dice      = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou       = (tp + eps)       / (tp + fp + fn + eps)
    precision = (tp + eps)       / (tp + fp + eps)
    recall    = (tp + eps)       / (tp + fn + eps)
    return {
        "dice":      float(dice),
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str = "cpu", threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate a segmentation model on a validation DataLoader.

    Runs a full pass over all batches, accumulates TP/FP/FN globally,
    then computes metrics at the end. No gradients are computed (@torch.no_grad).

    Args:
        model: the segmentation model (outputs raw logits, NOT probabilities)
        loader: DataLoader yielding (images, masks) batches
        device: 'cpu' or 'cuda'
        threshold: probability threshold to binarize predictions (default 0.5)
                   pixels with sigmoid(logit) >= 0.5 are predicted as tumor

    Returns:
        dict with keys: 'dice', 'iou', 'precision', 'recall'
    """
    model.eval()   # disable dropout and batchnorm training behavior
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        # Forward pass: model outputs logits (pre-sigmoid values)
        # Apply sigmoid to convert to probabilities in [0, 1]
        probs = torch.sigmoid(model(images))

        # Threshold probabilities to get binary predictions: 1 = tumor, 0 = background
        preds = (probs >= threshold).float()

        # Count confusion matrix components across all pixels in this batch
        # preds * masks            = correctly predicted tumor pixels (TP)
        # preds * (1 - masks)      = predicted tumor but actually background (FP)
        # (1 - preds) * masks      = missed tumor pixels (FN)
        tp_total += float((preds * masks).sum().item())
        fp_total += float((preds * (1.0 - masks)).sum().item())
        fn_total += float(((1.0 - preds) * masks).sum().item())

    # Compute final metrics from accumulated global counts
    return _metrics_from_counts(tp_total, fp_total, fn_total)
