from __future__ import annotations

# train.py — Main training script for 2D U-Net variants on BraTS data
#
# Supports all 3 model types (unet, attention_unet, hybrid) and 2 loss functions
# (dicebce, focal_tversky) via CLI arguments.
#
# Key feature: when multiple cases are found, automatically switches to
# patient-level split (train on one patient, validate on another) to avoid
# data leakage from slice-level splitting.
#
# Usage example:
#   python train.py --model-type attention_unet --loss dicebce --epochs 30 \
#                   --batch-size 4 --cpu --output-dir runs/attn_run

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from data import (
    build_dataset_from_cases, discover_case_paths,
    is_h5_dataset_dir, discover_h5_volume_ids, build_dataset_from_h5,
)
from dataset import BraTS2DSliceDataset, make_loaders
from eval import evaluate
from losses import DiceBCELoss, FocalTverskyLoss
from model import UNet2D, AttentionUNet2D, HybridUNet2D


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    """Run one full training epoch. Returns average loss over all batches."""
    model.train()
    running = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


def split_case_paths(case_paths: Sequence[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split patient paths into train/val groups (patient-level split).

    Ensures each patient is entirely in either train or val — never both.
    This prevents the data leakage that happens with slice-level splitting,
    where slices from the same patient appear in both sets.
    """
    paths = list(case_paths)
    if len(paths) < 2:
        return paths, []  # can't split with only 1 patient

    rng = random.Random(seed)
    rng.shuffle(paths)

    n_val   = max(1, int(len(paths) * val_ratio))
    n_train = len(paths) - n_val
    if n_train < 1:
        n_train, n_val = len(paths) - 1, 1

    return paths[:n_train], paths[n_train:]


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Factory: instantiate model based on --model-type CLI argument."""
    kwargs = dict(in_channels=4, out_channels=1, base_channels=args.base_channels)
    if args.model_type == "unet":
        return UNet2D(**kwargs)
    elif args.model_type == "attention_unet":
        return AttentionUNet2D(**kwargs)
    elif args.model_type == "hybrid":
        return HybridUNet2D(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def build_loss(args: argparse.Namespace) -> torch.nn.Module:
    """Factory: instantiate loss function based on --loss CLI argument."""
    if args.loss == "dicebce":
        return DiceBCELoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
    elif args.loss == "focal_tversky":
        return FocalTverskyLoss(alpha=args.ft_alpha, beta=args.ft_beta, gamma=args.ft_gamma)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def _build_loaders_from_h5(args: argparse.Namespace, only_tumor: bool):
    """Build train/val DataLoaders from BraTS 2020 HDF5 dataset (patient-level split)."""
    case_path = Path(args.case_path)
    all_vids  = discover_h5_volume_ids(case_path, max_cases=args.max_cases)
    print(f"HDF5 mode: {len(all_vids)} volumes found in {case_path}")

    if len(all_vids) < 2:
        X, y, summary = build_dataset_from_h5(case_path, all_vids, only_tumor, args.min_tumor_pixels)
        print(f"Single-volume mode: X={X.shape}")
        train_loader, val_loader = make_loaders(X, y, batch_size=args.batch_size,
                                                val_ratio=args.val_ratio, seed=args.seed)
        return train_loader, val_loader

    # Patient-level split: shuffle by volume ID
    rng = random.Random(args.seed)
    vids = list(all_vids)
    rng.shuffle(vids)
    n_val      = max(1, int(len(vids) * args.val_ratio))
    train_vids = vids[:-n_val]
    val_vids   = vids[-n_val:]
    print(f"Patient split: {len(train_vids)} train / {len(val_vids)} val volumes")

    X_train, y_train, train_summary = build_dataset_from_h5(
        case_path, train_vids, only_tumor, args.min_tumor_pixels)
    X_val,   y_val,   val_summary   = build_dataset_from_h5(
        case_path, val_vids,   only_tumor, args.min_tumor_pixels)

    print(f"Train: X={X_train.shape} | Val: X={X_val.shape}")
    for s in train_summary + val_summary:
        print(f"  {s['case_id']}: {s['num_slices']} tumor slices")

    train_ds     = BraTS2DSliceDataset(X_train, y_train)
    val_ds       = BraTS2DSliceDataset(X_val,   y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size,
                                               shuffle=False, num_workers=0)
    return train_loader, val_loader


def run_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    set_seed(args.seed)
    device  = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only_tumor = not args.include_empty_slices

    # ── Auto-detect dataset format ────────────────────────────────────────────
    case_path = Path(args.case_path)
    if case_path.is_dir() and is_h5_dataset_dir(case_path):
        print(f"Detected BraTS 2020 HDF5 format: {case_path}")
        train_loader, val_loader = _build_loaders_from_h5(args, only_tumor)
    else:
        # NIfTI format: BraTS 2021 (.tar) or BraTS 2024 (.tar.gz)
        case_paths = discover_case_paths(args.case_path, max_cases=args.max_cases)
        print(f"Detected NIfTI format. Discovered cases: {len(case_paths)}")
        for path in case_paths:
            print(f"  - {path}")

        if len(case_paths) == 1:
            # Single patient — slice-level split
            X, y, summary = build_dataset_from_cases(
                case_paths, only_tumor=only_tumor, min_tumor_pixels=args.min_tumor_pixels,
            )
            print(f"Single-case mode. Loaded slices: X={X.shape}, y={y.shape}")
            print(f"Case summary: {summary}")
            train_loader, val_loader = make_loaders(
                X, y, batch_size=args.batch_size, val_ratio=args.val_ratio,
                seed=args.seed, num_workers=0,
            )
        else:
            # Multiple patients — patient-level split
            train_cases, val_cases = split_case_paths(case_paths, args.val_ratio, args.seed)
            print(f"Patient split: train_cases={len(train_cases)}, val_cases={len(val_cases)}")

            X_train, y_train, train_summary = build_dataset_from_cases(
                train_cases, only_tumor=only_tumor, min_tumor_pixels=args.min_tumor_pixels,
            )
            print(f"Train slices: X={X_train.shape}, y={y_train.shape}")
            print(f"Train summary: {train_summary}")

            train_ds     = BraTS2DSliceDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
            )

            if val_cases:
                X_val, y_val, val_summary = build_dataset_from_cases(
                    val_cases, only_tumor=only_tumor, min_tumor_pixels=args.min_tumor_pixels,
                )
                print(f"Val slices: X={X_val.shape}, y={y_val.shape}")
                print(f"Val summary: {val_summary}")
                val_ds     = BraTS2DSliceDataset(X_val, y_val)
                val_loader = torch.utils.data.DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
                )
            else:
                val_loader = None

    # Optional: overfit on a tiny subset to verify the training loop works
    if args.tiny_overfit_samples > 0:
        subset_n = min(args.tiny_overfit_samples, len(train_loader.dataset))
        subset   = torch.utils.data.Subset(train_loader.dataset, list(range(subset_n)))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        print(f"Tiny overfit mode enabled with {subset_n} samples")

    model    = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} | Loss: {args.loss} | Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn   = build_loss(args)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_dice": [], "val_iou": [], "val_precision": [], "val_recall": []
    }
    best_dice           = -1.0
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device=device)
        else:
            metrics = {"dice": float("nan"), "iou": float("nan"),
                       "precision": float("nan"), "recall": float("nan")}

        history["val_dice"].append(metrics["dice"])
        history["val_iou"].append(metrics["iou"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_dice={metrics['dice']:.4f} val_iou={metrics['iou']:.4f}"
        )

        # Track best model and save checkpoint when val_dice improves
        score    = metrics["dice"] if val_loader is not None else -train_loss
        improved = score > (best_dice + args.early_stopping_min_delta)
        if improved:
            best_dice = score
            epochs_without_improve = 0
            best_path = out_dir / "best.pt"
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "best_dice": best_dice}, best_path)
        else:
            epochs_without_improve += 1

        # Early stopping: halt if no improvement for patience epochs
        if val_loader is not None and args.early_stopping_patience > 0:
            if epochs_without_improve >= args.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"no val_dice improvement > {args.early_stopping_min_delta} "
                    f"for {args.early_stopping_patience} epoch(s)."
                )
                break

    # Save latest checkpoint and full training history
    latest_path = out_dir / "latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": history}, latest_path)

    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved artifacts in: {out_dir.resolve()}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D binary U-Net baseline for BraTS slices.")
    parser.add_argument("--case-path", type=str,
                        default="BraTS2020_training_data/content/data",
                        help="Path to BraTS2020 HDF5 dir, a .tar case, or a dataset directory.")
    parser.add_argument("--output-dir", type=str, default="runs/baseline_2d")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--max-cases",   type=int,   default=0, help="Limit discovered cases (0 = all).")
    parser.add_argument("--min-tumor-pixels",    type=int, default=1)
    parser.add_argument("--include-empty-slices", action="store_true")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int,  default=16)
    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "attention_unet", "hybrid"],
                        help="Model architecture to use.")
    parser.add_argument("--loss", type=str, default="dicebce",
                        choices=["dicebce", "focal_tversky"],
                        help="Loss function.")
    parser.add_argument("--ft-alpha", type=float, default=0.7, help="Focal Tversky FP weight.")
    parser.add_argument("--ft-beta",  type=float, default=0.3, help="Focal Tversky FN weight.")
    parser.add_argument("--ft-gamma", type=float, default=0.75, help="Focal Tversky focal exponent.")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--bce-weight",  type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--tiny-overfit-samples", type=int, default=0,
                        help="Use only N samples to sanity-check that training loop works.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if val_dice does not improve for N epochs (0 = disabled).")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum improvement to reset early stopping patience.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
