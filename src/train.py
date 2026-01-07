from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.mnist_data import MNISTConfig, get_mnist_dataloaders
from src.models.cnn_mnist import MNISTCNN
from src.training.engine import evaluate, fit
from src.utils.device import get_device
from src.utils.paths import artifacts_dir
from src.utils.seed import set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 128


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = get_device()
    print(f"Using device: {device}")

    # Data
    data_cfg = MNISTConfig(batch_size=cfg.batch_size, seed=cfg.seed)
    train_loader, val_loader, test_loader = get_mnist_dataloaders(data_cfg)

    # Model
    model = MNISTCNN().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Train
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.epochs,
    )

    # Test
    test_res = evaluate(model, test_loader, device)
    print(f"\nTEST  | loss {test_res.loss:.4f} acc {test_res.accuracy:.4f}")

    # Save artifact
    out_dir = artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "mnist_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "MNISTCNN",
            "normalize_mean": (0.1307,),
            "normalize_std": (0.3081,),
        },
        weights_path,
    )
    print(f"\nSaved model weights to: {weights_path}")


if __name__ == "__main__":
    main()
