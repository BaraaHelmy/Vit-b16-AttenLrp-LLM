# scripts/inspect_lrp_predictions.py
# Reads LRP_results.json and outputs the confidence scores for the LRP samples
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

    # Dataset (TEST SET - same as used for LRP generation)
    dataset = CUBDataset(
        root=str(data_root),
        split="test",  # Using test set
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

    print("\n" + "=" * 80)
    print("New Confidence Results (Test Set)")
    print("=" * 80)
    print(f"{'Sample':<10} {'GT':<30} {'PRED':<30} {'Confidence':<12} {'Status':<10}")
    print("-" * 80)

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for r in results:
            idx = r["sample_idx"]
            total_count += 1

            x, gt = dataset[idx]
            x = x.unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            conf, pred = probs.max(dim=1)

            gt = int(gt)
            pred = int(pred)
            conf = float(conf)

            # Determine if prediction is correct
            is_correct = (gt == pred)
            if is_correct:
                correct_count += 1
                status = "✓ Correct"
            else:
                status = "✗ Wrong"

            # Format class names (replace underscores with spaces for readability)
            gt_name = id_to_name[gt].replace("_", " ")
            pred_name = id_to_name[pred].replace("_", " ")

            print(
                f"{idx:<10} "
                f"{gt_name:<30} "
                f"{pred_name:<30} "
                f"{conf:<12.3f} "
                f"{status:<10}"
            )

    print("-" * 80)
    print(f"\nSummary: {correct_count}/{total_count} correct ({100*correct_count/total_count:.1f}% accuracy)")
    print("=" * 80)
    print("\nDone.")


if __name__ == "__main__":
    main()
