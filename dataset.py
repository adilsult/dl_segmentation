from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BraTS2DSliceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (N, C, H, W), got {X.shape}")
        if y.ndim != 4:
            raise ValueError(f"Expected y shape (N, 1, H, W), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset = BraTS2DSliceDataset(X, y)
    n_total = len(dataset)
    if n_total < 2:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, None

    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train < 1:
        n_train, n_val = n_total - 1, 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

