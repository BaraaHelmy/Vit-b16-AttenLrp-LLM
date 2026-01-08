# scripts/inspect_lrp_predictions.py
#
# Inspect predictions for the 20 LRP samples:
# - dataset index
# - ground truth class (id + name)
# - predicted class (id + name)
# - confidence

from pathlib import Path
import json
import torch

from cub_dataset import CUBDataset
from vit_cub_model import create_vit_cub_model


def load_class_names(classes_txt):
    """Map 0-based class index -> CUB class name"""
    id_to_name = {}
    with open(classes_txt, "r") as f:
        for line in f:
            cid, name = line.strip().split()
            id_to_name[int(cid) - 1] = name  # convert to 0-based
    return id_to_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    data_root = Path("dataset/raw/CUB_200_2011")
    results_path = Path("results/lrp_samples/lrp_results.json")
    ckpt_path = Path("checkpoints/vit_cub_best.pt")
    classes_txt = data_root / "classes.txt"

    # Dataset
    dataset = CUBDataset(
        root=str(data_root),
        split="test",
        transform=None,
    )

    # Model
    model = create_vit_cub_model(
        num_classes=200,
        pretrained=False,
        freeze_backbone=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Class names
    id_to_name = load_class_names(classes_txt)

    # LRP results
    results = json.load(open(results_path))

    print("\n=== Predictions for LRP samples ===\n")

    with torch.no_grad():
        for r in results:
            idx = r["sample_idx"]

            x, gt = dataset[idx]
            x = x.unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            conf, pred = probs.max(dim=1)

            gt = int(gt)
            pred = int(pred)
            conf = float(conf)

            print(
                f"idx={idx:6d} | "
                f"GT={gt:3d} ({id_to_name[gt]}) | "
                f"PRED={pred:3d} ({id_to_name[pred]}) | "
                f"conf={conf:.3f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
