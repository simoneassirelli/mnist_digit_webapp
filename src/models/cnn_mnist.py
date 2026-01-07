from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    Convolutional Neural Network for handwritten digit classification (MNIST).

    Architecture overview:
    - Two convolutional layers extract local visual patterns (edges, curves).
    - Max pooling reduces spatial resolution and increases robustness.
    - Fully connected layers combine extracted features to predict the digit.

    Input:
        x (torch.Tensor): Batch of grayscale images with shape [B, 1, 28, 28]

    Output:
        torch.Tensor: Logits with shape [B, 10], one score per digit class (0–9).
        Softmax is applied externally when probabilities are needed.
    """

    def __init__(self) -> None:
        """
        Initialize the MNIST CNN layers.

        Layers:
        - conv1: Extracts low-level features from raw pixels (edges, strokes).
        - conv2: Combines low-level features into higher-level shapes.
        - pool: Downsamples feature maps by a factor of 2.
        - dropout: Regularization to reduce overfitting.
        - fc1: Dense layer combining all extracted features.
        - fc2: Final classification layer producing class logits.
        """
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )  # -> [B, 32, 28, 28]

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )  # -> [B, 64, 28, 28]

        # Spatial downsampling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )  # -> [B, 64, 14, 14]

        # Regularization
        self.dropout = nn.Dropout(p=0.25)

        # Fully connected classifier
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Steps:
        1. Apply first convolution + ReLU activation.
        2. Apply second convolution + ReLU activation.
        3. Downsample feature maps using max pooling.
        4. Apply dropout for regularization.
        5. Flatten feature maps into a 1D vector.
        6. Apply fully connected layer + ReLU.
        7. Apply dropout.
        8. Produce class logits.

        Args:
            x (torch.Tensor): Input batch of images with shape [B, 1, 28, 28].

        Returns:
            torch.Tensor: Logits with shape [B, 10].
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 14, 14]

        # Regularization
        x = self.dropout(x)

        # Flatten spatial dimensions (image → vector) Fully connected layers expect vectors, not images
        x = torch.flatten(x, start_dim=1)  # [B, 64 * 14 * 14]

        # Fully connected classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output raw class scores (logits)
        return self.fc2(x)
