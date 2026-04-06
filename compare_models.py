"""Train hybrid model configurations on combined BraTS cases with slice-level split.

UNet and AttentionUNet results already exist from prior run.
This script trains only the remaining Hybrid configs (lighter transformer).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from data import build_dataset_from_cases, discover_case_paths
from dataset import BraTS2DSliceDataset, make_loaders
from eval import evaluate
from losses import DiceBCELoss, FocalTverskyLoss
from model import UNet2D, AttentionUNet2D, HybridUNet2D
from train import set_seed, train_one_epoch

CONFIGS = [
    {"name": "hybrid_dicebce",        "model": "hybrid",         "loss": "dicebce"},
    {"name": "hybrid_focal_tversky",  "model": "hybrid",         "loss": "focal_tversky"},
]

EPOCHS = 30
BATCH_SIZE = 4
VAL_RATIO = 0.2
SEED = 42
BASE_CHANNELS = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
MIN_DELTA = 0.001


def build_model(name: str) -> torch.nn.Module:
    kw = dict(in_channels=4, out_channels=1, base_channels=BASE_CHANNELS)
    if name == "unet":
        return UNet2D(**kw)
    elif name == "attention_unet":
        return AttentionUNet2D(**kw)
    elif name == "hybrid":
        # Lighter transformer for CPU: 2 layers, smaller FFN
        return HybridUNet2D(**kw, num_heads=4, num_transformer_layers=2,
                            dim_feedforward=256, transformer_dropout=0.1)
    raise ValueError(name)


def build_loss(name: str) -> torch.nn.Module:
    if name == "dicebce":
        return DiceBCELoss()
    elif name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    raise ValueError(name)


def main() -> None:
    set_seed(SEED)
    device = "cpu"

    # Build combined dataset from all discovered cases
    case_paths = discover_case_paths(".", max_cases=0)
    print(f"Cases: {len(case_paths)}")
    for p in case_paths:
        print(f"  {p}")

    X, y, summary = build_dataset_from_cases(case_paths, only_tumor=True, min_tumor_pixels=1)
    print(f"Combined dataset: X={X.shape}, y={y.shape}")
    print(f"Summary: {summary}")

    # Create shared train/val loaders (slice-level split, same for all models)
    train_loader, val_loader = make_loaders(
        X, y, batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, seed=SEED, num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results = {}

    # Load existing results from prior runs (unet, attn)
    for prior_name in ["unet_dicebce", "attn_dicebce"]:
        hist_path = Path(f"runs/{prior_name}/history.json")
        if hist_path.exists():
            h = json.loads(hist_path.read_text())
            best_d = max(h["val_dice"])
            results[prior_name] = {
                "model": prior_name.replace("_dicebce", ""),
                "loss": "dicebce",
                "params": "—",
                "best_dice": best_d,
                "best_iou": max(h["val_iou"]),
                "best_epoch": h["val_dice"].index(best_d) + 1,
                "final_train_loss": h["train_loss"][-1],
                "elapsed_sec": 0,
            }
            print(f"Loaded prior result: {prior_name} best_dice={best_d:.4f}", flush=True)

    for cfg in CONFIGS:
        print(f"\n{'='*60}", flush=True)
        print(f"Training: {cfg['name']} (model={cfg['model']}, loss={cfg['loss']})", flush=True)
        print(f"{'='*60}", flush=True)

        set_seed(SEED)
        model = build_model(cfg["model"]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = build_loss(cfg["loss"])

        history = {"train_loss": [], "val_dice": [], "val_iou": [], "val_precision": [], "val_recall": []}
        best_dice = -1.0
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            metrics = evaluate(model, val_loader, device=device)

            history["train_loss"].append(train_loss)
            history["val_dice"].append(metrics["dice"])
            history["val_iou"].append(metrics["iou"])
            history["val_precision"].append(metrics["precision"])
            history["val_recall"].append(metrics["recall"])

            improved = metrics["dice"] > (best_dice + MIN_DELTA)
            if improved:
                best_dice = metrics["dice"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"  Epoch {epoch:02d}/{EPOCHS} "
                f"loss={train_loss:.4f} "
                f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
                f"{'*' if improved else ''}",
                flush=True,
            )

            if PATIENCE > 0 and epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0
        best_epoch = history["val_dice"].index(max(history["val_dice"])) + 1

        results[cfg["name"]] = {
            "model": cfg["model"],
            "loss": cfg["loss"],
            "params": n_params,
            "best_dice": best_dice,
            "best_iou": max(history["val_iou"]),
            "best_epoch": best_epoch,
            "final_train_loss": history["train_loss"][-1],
            "elapsed_sec": round(elapsed, 1),
            "history": history,
        }

        # Save per-model history
        out_dir = Path(f"runs/{cfg['name']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Params':>10} {'Best Dice':>10} {'Best IoU':>10} {'Epoch':>6} {'Time':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<25} {r['params']:>10,} {r['best_dice']:>10.4f} {r['best_iou']:>10.4f} "
            f"{r['best_epoch']:>6} {r['elapsed_sec']:>7.0f}s"
        )

    # Save summary
    summary_path = Path("runs/comparison_summary.json")
    summary_data = {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in results.items()}
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
