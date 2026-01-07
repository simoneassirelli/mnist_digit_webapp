from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float

# Prevents PyTorch from storing computation graphs. This:
# - speeds up inference
# - reduces memory
# - guarantees no gradients are computed
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EpochResult:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0) # loss.item() is the average loss over the batch
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return EpochResult(loss=total_loss / total, accuracy=correct / total) # So to compute epoch loss correctly, we multiply by batch size and then divide by total at the end


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochResult:

    # Enables training behavior:
    # dropout active
    # batchnorm updates (if present)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # clears old gradients
        optimizer.zero_grad(set_to_none=True) 

        # forward pass → logits
        logits = model(x)

        # compute loss
        loss = criterion(logits, y)

        # computes gradients
        loss.backward()

        # update weights
        # The update happens once per batch at optimizer.step()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return EpochResult(loss=total_loss / total, accuracy=correct / total)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> Dict[str, list]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, optimizer, device)
        val_res = evaluate(model, val_loader, device)

        history["train_loss"].append(train_res.loss)
        history["train_acc"].append(train_res.accuracy)
        history["val_loss"].append(val_res.loss)
        history["val_acc"].append(val_res.accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_res.loss:.4f} acc {train_res.accuracy:.4f} | "
            f"val loss {val_res.loss:.4f} acc {val_res.accuracy:.4f}"
        )

    return history
