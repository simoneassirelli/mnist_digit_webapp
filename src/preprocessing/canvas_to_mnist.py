from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


def _center_of_mass(img: np.ndarray) -> Tuple[float, float]:
    # img is 2D, values 0..255 where higher means more ink
    total = img.sum()
    if total == 0:
        return (img.shape[0] / 2.0, img.shape[1] / 2.0)
    ys, xs = np.indices(img.shape)
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total
    return cy, cx


def canvas_png_bytes_to_mnist_tensor(
    png_bytes: bytes,
    normalize_mean: Tuple[float, ...] = (0.1307,),
    normalize_std: Tuple[float, ...] = (0.3081,),
) -> torch.Tensor:
    """
    Convert a canvas PNG (white background, dark stroke) into a MNIST-like tensor:
    - grayscale
    - invert so digit is "white" on black (MNIST-like)
    - crop to bounding box of ink
    - resize to 20x20 and pad to 28x28
    - center by center of mass
    - scale to [0,1] then normalize
    Returns: torch.Tensor shape [1, 1, 28, 28]
    """

    # Load image
    img = Image.open(io.BytesIO(png_bytes)).convert("L")  # grayscale

    # Invert: make ink high values (MNIST-like: digit bright on dark bg)
    img = ImageOps.invert(img)

    # Convert to numpy for cropping
    arr = np.array(img)

    # Threshold to find bounding box (very light noise ignored)
    mask = arr > 20
    if not mask.any():
        # empty drawing -> return zeros
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        # normalize like training
        x = (x - normalize_mean[0]) / normalize_std[0]
        return x

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Crop
    img_cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Resize so the longer side becomes 20 pixels, keeping aspect ratio
    w, h = img_cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20 / h))))
    img_resized = img_cropped.resize((new_w, new_h), resample=Image.BILINEAR)

    # Paste into 28x28 canvas (centered initially)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(img_resized, (left, top))

    # Center by center of mass (MNIST-style)
    arr28 = np.array(canvas).astype(np.float32)
    cy, cx = _center_of_mass(arr28)
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))

    shifted = np.roll(arr28, shift=shift_y, axis=0)
    shifted = np.roll(shifted, shift=shift_x, axis=1)

    # Convert to tensor in [0,1]
    x = torch.tensor(shifted / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Normalize
    x = (x - normalize_mean[0]) / normalize_std[0]
    return x
