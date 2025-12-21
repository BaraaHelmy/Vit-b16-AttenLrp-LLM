# scripts/train_vit_cub.py

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from cub_dataset import CUBDataset
from vit_cub_model import create_vit_cub_model


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CUB-200.
    We use the same transforms as already defined in CUBDataset.
    """

    train_dataset = CUBDataset(
        root=str(data_root),
        split="train",
        transform=None,  # CUBDataset handles ViT transforms internally
    )

    test_dataset = CUBDataset(
        root=str(data_root),
        split="test",
        transform=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches=None,  # optional for quick tests
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    Returns: (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)  # [B, num_classes]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # progress print every 10 batches
        if batch_idx % 10 == 0:
            print(f"  [batch {batch_idx+1}/{num_batches}] " f"loss: {loss.item():.4f}")

        # OPTIONAL: limit number of batches for quick tests
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model.
    Returns: (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    # ============================
    # 1. Basic config
    # ============================
    data_root = Path("dataset/raw/CUB_200_2011")  # adjust if needed
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 200
    batch_size = 32
    num_workers = 4
    num_epochs = 40
    lr = 1e-4
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================
    # 2. Data
    # ============================
    train_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ============================
    # 3. Model, loss, optimizer
    # ============================
    model = create_vit_cub_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False,  # change to True if you want to freeze later
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0

    # ============================
    # 4. Training loop
    # ============================
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=None,  # run full epoch
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Train  - loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%\n"
            f"Val    - loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoints_dir / "vit_cub_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(
                f"-> Saved new best model to {ckpt_path} (val_acc={val_acc*100:.2f}%)"
            )

    print("\nTraining finished.")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
