from __future__ import annotations

from src.data.mnist_data import MNISTConfig, download_mnist, get_mnist_dataloaders


def main() -> None:
    print("Downloading MNIST (if needed)...")
    download_mnist()

    cfg = MNISTConfig(batch_size=128, num_workers=2)
    train_loader, val_loader, test_loader = get_mnist_dataloaders(cfg)

    x, y = next(iter(train_loader))
    print("Sanity check batch:")
    print(f"  x.shape = {x.shape}  (expected: [B, 1, 28, 28])")
    print(f"  y.shape = {y.shape}  (expected: [B])")
    print(f"  y[:10]  = {y[:10].tolist()}")

    print("\nCounts:")
    print(f"  train batches: {len(train_loader)}")
    print(f"  val batches:   {len(val_loader)}")
    print(f"  test batches:  {len(test_loader)}")


if __name__ == "__main__":
    main()
