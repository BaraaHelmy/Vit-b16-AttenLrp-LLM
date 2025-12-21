"""
Small sanity check for the CUB Dataset + DataLoader.

What it does:
  - Instantiates CUBDataset for train and test splits
  - Wraps them in DataLoaders
  - Prints dataset sizes
  - Fetches one batch from the train loader and prints tensor shapes
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.cub_dataset import CUBDataset

def inspect_some_samples(dataset, num_samples: int = 5) -> None:
    """
    Print a few (path, label) pairs from the dataset to verify that
    paths and labels look reasonable.
    """
    print(f"\n=== Inspecting first {num_samples} samples from {dataset.split} dataset ===")
    for i in range(num_samples):
        img_path, label = dataset.samples[i]
        print(f"Sample {i}:")
        print(f"  path  = {img_path}")
        print(f"  label = {label}")

def main() -> None:
    # Path to the CUB_200_2011 root folder.
    # This folder should contain: images.txt, image_class_labels.txt,
    # train_test_split.txt, and the images/ directory.
    dataset_root = Path("dataset/raw/CUB_200_2011")

    # 1) Instantiate train and test datasets
    train_dataset = CUBDataset(root=dataset_root, split="train")
    test_dataset = CUBDataset(root=dataset_root, split="test")

    print("=== CUB Dataset Info ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Num classes:   {train_dataset.num_classes}")

        # Optional: inspect a few samples from the train split
    inspect_some_samples(train_dataset, num_samples=5)


    # 2) Wrap in DataLoaders
    batch_size = 32  # small, safe batch size for CPU sanity check

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,   # if this causes issues, we'll set it to 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # 3) Get one batch from the train loader
    print("\n=== One train batch ===")
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"images.shape = {images.shape}")   # expect: [B, 3, 224, 224]
    print(f"labels.shape = {labels.shape}")   # expect: [B]
    print(f"images dtype = {images.dtype}")
    print(f"labels dtype = {labels.dtype}")

    print("\nSanity check passed if:")
    print("  - shapes make sense")
    print("  - no FileNotFoundError or other crashes occurred")


if __name__ == "__main__":
    main()
