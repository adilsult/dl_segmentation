from __future__ import annotations

from typing import Dict

import torch


def _metrics_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-6) -> Dict[str, float]:
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str = "cpu", threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        probs = torch.sigmoid(model(images))
        preds = (probs >= threshold).float()

        tp_total += float((preds * masks).sum().item())
        fp_total += float((preds * (1.0 - masks)).sum().item())
        fn_total += float(((1.0 - preds) * masks).sum().item())

    return _metrics_from_counts(tp_total, fp_total, fn_total)

