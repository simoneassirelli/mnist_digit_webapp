from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.paths import mnist_dir


@dataclass(frozen=True)
class MNISTConfig:
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    val_size: int = 10_000  # validation split from training set
    seed: int = 42


def download_mnist() -> None:
    """
    Downloads MNIST into ./data/mnist (if not already present).
    """
    root = mnist_dir()
    root.mkdir(parents=True, exist_ok=True)

    # No transform needed for download; torchvision handles files.
    datasets.MNIST(root=str(root), train=True, download=True)
    datasets.MNIST(root=str(root), train=False, download=True)


def get_mnist_dataloaders(
    cfg: MNISTConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader)

    Notes:
    - Uses ToTensor() which scales pixels to [0, 1].
    - Normalizes using MNIST mean/std for better training stability.
    """
    root = mnist_dir()
    root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0,255] -> [0,1], shape: [1,28,28]
            transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
        ]
    )

    full_train = datasets.MNIST(
        root=str(root),
        train=True,
        download=True,   # safe: no-op if already downloaded
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )

    # Split train into train/val deterministically
    torch.manual_seed(cfg.seed)
    train_size = len(full_train) - cfg.val_size
    if train_size <= 0:
        raise ValueError(f"val_size={cfg.val_size} is too large for MNIST.")

    train_ds, val_ds = random_split(full_train, [train_size, cfg.val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader
