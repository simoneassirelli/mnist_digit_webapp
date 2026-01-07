from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from src.models.cnn_mnist import MNISTCNN
from src.utils.device import get_device
from src.utils.paths import artifacts_dir


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    normalize_mean: Tuple[float, ...]
    normalize_std: Tuple[float, ...]


def load_mnist_model(weights_path: Path | None = None) -> LoadedModel:
    device = get_device() # So if you deploy to GPU someday, it will use GPU automatically; otherwise CPU.

    if weights_path is None:
        weights_path = artifacts_dir() / "mnist_cnn.pt"

    ckpt = torch.load(weights_path, map_location=device)

    # Instantiate architecture + load weights
    model = MNISTCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # This is the canonical PyTorch inference pattern:
    # define the model code
    # load trained weights
    # switch to eval mode

    # Return normalization stats too
    mean = tuple(ckpt.get("normalize_mean", (0.1307,)))
    std = tuple(ckpt.get("normalize_std", (0.3081,)))

    return LoadedModel(model=model, device=device, normalize_mean=mean, normalize_std=std)
