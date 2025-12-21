# scripts/test_vit_forward.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cub_dataset import CUBDataset          # same as before
from vit_cub_model import create_vit_cub_model  # NEW import


def main():
    data_root = Path("dataset/raw/CUB_200_2011")

    batch_size = 32
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CUBDataset(
        root=str(data_root),
        split="train",
        transform=None,  # keep your existing ViT transforms from CUBDataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # === use our CUB-adapted ViT ===
    model = create_vit_cub_model(
        num_classes=200,
        pretrained=True,
        freeze_backbone=False,  # we decide later if we want to freeze
    )

    model.to(device)
    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)  # should be [B, 200]

    print(f"Outputs shape: {outputs.shape}")  # expect [32, 200]

    probs = torch.softmax(outputs, dim=1)
    top5_prob, top5_idx = torch.topk(probs[0], k=5)
    print("Top-5 predicted CUB logits indices for sample 0:")
    print("indices:", top5_idx.cpu().numpy())
    print("probs:  ", top5_prob.cpu().numpy())

    print("Forward pass with 200-class head completed successfully.")


if __name__ == "__main__":
    main()
