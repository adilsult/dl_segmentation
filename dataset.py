from __future__ import annotations

# dataset.py — PyTorch Dataset and DataLoader wrappers for BraTS 2D slices
#
# This module bridges the raw numpy arrays from data.py and PyTorch's training loop.
# It provides:
# - BraTS2DSliceDataset: a PyTorch Dataset that wraps (X, y) numpy arrays
# - make_loaders: creates train/val DataLoaders with a reproducible random split
#
# All slices from data.py are already 2D axial slices with shape (4, 240, 240),
# so no additional preprocessing is needed here — just type conversion to tensors.

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BraTS2DSliceDataset(Dataset):
    """PyTorch Dataset that wraps pre-extracted BraTS 2D slices.

    Stores all slices in memory as float32 tensors for fast access during training.

    Args:
        X: numpy array of shape (N, 4, 240, 240) — 4-channel MRI input slices
        y: numpy array of shape (N, 1, 240, 240) — binary tumor mask targets
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate shapes before converting — fail fast with a clear error message
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (N, C, H, W), got {X.shape}")
        if y.ndim != 4:
            raise ValueError(f"Expected y shape (N, 1, H, W), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}")

        # Convert numpy arrays to PyTorch tensors once at init time
        # This avoids repeated conversion during training which would slow down the dataloader
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        # PyTorch needs this to know the size of the dataset
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns a single (image, mask) pair for a given slice index
        # The DataLoader calls this repeatedly to build batches
        return self.X[index], self.y[index]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create reproducible train and validation DataLoaders from (X, y) arrays.

    Performs a random 80/20 slice-level split (by default) using a fixed seed
    so results are reproducible across runs.

    NOTE: This is a slice-level split — slices from the same patient can appear
    in both train and val. This is fast to set up but inflates val Dice because
    the model sees the same patient's anatomy during training.
    For honest evaluation, use split_case_paths() in train.py instead (patient-level split).

    Args:
        X: (N, 4, 240, 240) float32 input array
        y: (N, 1, 240, 240) float32 mask array
        batch_size: number of slices per batch
        val_ratio: fraction of slices to use for validation (default: 0.2 = 20%)
        seed: random seed for reproducible splits
        num_workers: parallel data loading workers (0 = main process, safe for small datasets)

    Returns:
        train_loader, val_loader — (val_loader is None if only 1 sample available)
    """
    dataset = BraTS2DSliceDataset(X, y)
    n_total = len(dataset)

    # Edge case: only 1 sample — can't split, return everything as training
    if n_total < 2:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, None

    # Calculate split sizes, ensuring at least 1 sample in each split
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train < 1:
        n_train, n_val = n_total - 1, 1

    # Use a seeded generator so the same split is produced every run
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    # shuffle=True for training (randomize order each epoch), False for validation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
