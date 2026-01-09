# scripts/test_vit_forward.py
#
# Purpose:
# - Generates LRP heatmaps/visualizations and saves which sample indices were selected to lrp_results.json
# - Sanity-check ViT forward pass on CUB
# - Enable AttnLRP/CP-LRP via LXT monkey-patching for torchvision ViT
# - Compute pixel-level relevance for MULTIPLE images via backward()
# - Save for each sample:
#     1) lrp_heatmap_XXX.png         (heatmap only, from model-view image)
#     2) lrp_overlay_XXX.png         (model-view image + heatmap overlay)
#     3) lrp_comparison_XXX.png      (3-panel: model-view image | heatmap | overlay)
#     4) lrp_relevance_raw_XXX.pt    (raw relevance tensor)
# - Save once:
#     5) lrp_image_paths.txt         (all original file paths)
#     6) lrp_results.json            (metadata JSON with gamma parameters)
#
# GAMMA RULE ENHANCEMENT:
# - Vision Transformers are susceptible to "gradient shattering" which leads to noisy heatmaps.
# - The Gamma rule from zennit improves signal-to-noise ratio by emphasizing positive relevance.
# - Higher gamma values = more positive emphasis, less noise (but may oversuppress negative contributions).
# - Gamma only affects the backward pass (LRP computation), NOT the forward pass (predictions).
# - Recommended starting values: conv_gamma=1.0, lin_gamma=0.5 (balanced noise reduction).
#
# IMPORTANT FIX (thesis-grade correctness):
# - The "Original Image" shown in comparison is now the *exact model-view image*
#   (post resize/crop, pre-normalization reversed for display), NOT a naive PIL resize.
#   This ensures the heatmap aligns honestly with what the model actually saw.

from pathlib import Path
import json
import random

import numpy as np
import torch
from PIL import Image  # kept (we store original paths; no longer used for resizing)
from torch.utils.data import DataLoader

from cub_dataset import CUBDataset
from vit_cub_model import create_vit_cub_model

# ------------------------------------------------------------------
# LXT (LRP-eXplains-Transformers) for torchvision ViT in THIS repo:
# It works via monkey-patching torchvision.models.vision_transformer.
# This MUST happen before creating the ViT model.
# ------------------------------------------------------------------
from torchvision.models import vision_transformer
from lxt.efficient import monkey_patch, monkey_patch_zennit

monkey_patch(vision_transformer, verbose=True)  # Enables LRP rules in backward pass

# ------------------------------------------------------------------
# Zennit Integration for Gamma Rule:
# - zennit provides the Gamma rule which LXT doesn't natively support.
# - monkey_patch_zennit transforms zennit's explicit LRP formulation into
#   the Input*Gradient formulation compatible with LXT.
# - This enables noise reduction and positive relevance emphasis for ViT heatmaps.
# ------------------------------------------------------------------
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

monkey_patch_zennit(verbose=True)  # Make zennit compatible with LXT


def compute_lrp_for_sample(model, dataset, sample_idx, device, conv_gamma=1.0, lin_gamma=0.5):
    """
    Compute LRP relevance for a single sample using the Gamma rule.

    The Gamma rule is an LRP propagation rule that emphasizes positive contributions
    and reduces noise in the heatmaps. It modifies the relevance propagation formula:
        R_j = sum_k (a_j * (w_jk + gamma * max(0, w_jk))) / (sum_i a_i * (w_ik + gamma * max(0, w_ik))) * R_k
    
    Higher gamma values:
        - Increase emphasis on positive relevance (red regions in heatmaps)
        - Reduce noise and scattered artifacts
        - May oversuppress negative contributions (blue regions)
    
    IMPORTANT: The Gamma rule only affects the backward pass (relevance computation).
    The forward pass (model predictions, confidence scores) remains UNCHANGED.

    Args:
        model: The ViT model (must be monkey-patched with LXT)
        dataset: CUBDataset instance
        sample_idx: Index of the sample in the dataset
        device: torch device ('cuda' or 'cpu')
        conv_gamma: Gamma parameter for Conv2d layers (default: 1.0)
                    - Higher values = more positive emphasis, less noise
                    - Typical range: 0.1 to 100.0
                    - Recommended: 0.5-10.0 for balanced results
        lin_gamma: Gamma parameter for Linear layers (default: 0.5)
                   - Higher values = more positive emphasis, less noise
                   - Typical range: 0.0 to 1.0
                   - Recommended: 0.1-0.5 for balanced results

    Returns:
        dict with:
          - img_path (str): original file path on disk
          - label (int): ground-truth label from metadata
          - target_class (int): predicted class index (argmax) - UNCHANGED by gamma
          - relevance (torch.Tensor CPU): [1, 3, 224, 224] - IMPROVED by gamma
          - x (torch.Tensor CPU): model input tensor (normalized) [1, 3, 224, 224]
          - gamma_params (dict): {'conv_gamma': float, 'lin_gamma': float}
    """
    # dataset.samples holds (img_path, label) entries
    img_path, label = dataset.samples[sample_idx]

    # dataset[sample_idx] returns the transformed tensor the model sees
    x_tensor, _ = dataset[sample_idx]
    x = x_tensor.unsqueeze(0).to(device).requires_grad_(True)  # [1, 3, 224, 224]

    # Clear gradients/relevance from any previous computation
    model.zero_grad(set_to_none=True)
    x.grad = None  # Explicitly clear input gradients

    # ------------------------------------------------------------------
    # Define Gamma rules for Conv2d and Linear layers using zennit
    # LayerMapComposite maps specific layer types to specific LRP rules
    # ------------------------------------------------------------------
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),   # Patch embedding conv
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),   # Attention & MLP layers
    ])

    # Register the composite rules with the model
    # This installs backward hooks that modify gradient computation for LRP
    zennit_comp.register(model)

    try:
        # Forward pass - predictions are UNCHANGED by Gamma rule
        logits = model(x)  # [1, 200]
        target_class = logits.argmax(dim=1).item()

        # Backward pass - this is where Gamma rule improves the heatmap quality
        # The Gamma rule modifies how relevance is propagated through each layer
        logits[0, target_class].backward()

        # Pixel-level relevance (LRP-style) - now enhanced with Gamma rule
        # The input gradient represents the relevance of each input pixel
        relevance = x.grad  # [1, 3, 224, 224]

    finally:
        # Always remove the composite to prevent interference in future computations
        # This is critical when processing multiple samples
        zennit_comp.remove()

    return {
        "img_path": str(img_path),
        "label": int(label),
        "target_class": int(target_class),
        "relevance": relevance.detach().cpu(),
        "x": x.detach().cpu(),
        "gamma_params": {
            "conv_gamma": conv_gamma,
            "lin_gamma": lin_gamma,
        },
    }


def _to_model_view_uint8(x_cpu: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized model input tensor x (CPU) to a displayable uint8 RGB image.
    Assumes normalization: mean=0.5, std=0.5 per channel (i.e., x in ~[-1, 1]).

    Args:
        x_cpu: [1, 3, 224, 224] CPU tensor

    Returns:
        np.ndarray uint8 [224, 224, 3]
    """
    img_disp = x_cpu[0]  # [3, 224, 224]
    img_disp = (img_disp * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    img_disp = (img_disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HWC uint8
    return img_disp


def save_visualizations(lrp_result, sample_idx, output_dir: Path):
    """
    Save heatmap, overlay, comparison panel, and raw relevance tensor for a sample.
    Filenames are based on the dataset index (sample_idx), not the loop counter.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_str = f"{sample_idx:06d}"  # wider padding to avoid collisions / confusion

    relevance = lrp_result["relevance"]  # [1, 3, 224, 224] CPU
    x = lrp_result["x"]                  # [1, 3, 224, 224] CPU (normalized)
    img_path = lrp_result["img_path"]

    # --- Model-view "original" (truthful) ---
    orig_model_view = _to_model_view_uint8(x)  # [224, 224, 3] uint8

    # --- Heatmap: sum over channels -> 2D ---
    heat = relevance[0].sum(dim=0)  # [224, 224]
    heat = heat / (heat.abs().max() + 1e-8)  # normalize to [-1, 1] for visualization only
    heat_np = heat.numpy()

    # 1) Heatmap-only PNG
    plt.figure(figsize=(8, 8))
    plt.imshow(heat_np, cmap="bwr", vmin=-1, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"lrp_heatmap_{idx_str}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Overlay PNG (model-view image + heatmap)
    plt.figure(figsize=(8, 8))
    plt.imshow(orig_model_view)
    plt.imshow(heat_np, cmap="bwr", vmin=-1, vmax=1, alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"lrp_overlay_{idx_str}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Save raw relevance tensor (research artifact)
    torch.save(relevance, output_dir / f"lrp_relevance_raw_{idx_str}.pt")

    # 4) Comparison image: model-view image | heatmap | overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_model_view)
    axes[0].set_title("Model-view Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heat_np, cmap="bwr", vmin=-1, vmax=1)
    axes[1].set_title("LRP Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(orig_model_view)
    axes[2].imshow(heat_np, cmap="bwr", vmin=-1, vmax=1, alpha=0.45)
    axes[2].set_title("Overlay", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"lrp_comparison_{idx_str}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "sample_idx": int(sample_idx),
        "img_path": img_path,
        "label": int(lrp_result["label"]),
        "target_class": int(lrp_result["target_class"]),
        "files": {
            "heatmap": f"lrp_heatmap_{idx_str}.png",
            "overlay": f"lrp_overlay_{idx_str}.png",
            "comparison": f"lrp_comparison_{idx_str}.png",
            "relevance_raw": f"lrp_relevance_raw_{idx_str}.pt",
        },
    }


def main():
    data_root = Path("dataset/raw/CUB_200_2011")
    output_dir = Path("results/lrp_samples")
    num_samples = 20  # number of random samples to generate

    # ------------------------------------------------------------------
    # Gamma Rule Configuration
    # These parameters control the noise reduction and positive relevance emphasis.
    # Adjust these values to tune heatmap quality:
    #   - Higher values = more positive emphasis, less noise
    #   - Lower values = more balanced positive/negative, potentially more noise
    #
    # Recommended configurations:
    #   Conservative (moderate):     conv_gamma=0.5,  lin_gamma=0.1
    #   Balanced (recommended):      conv_gamma=1.0,  lin_gamma=0.5
    #   Aggressive (strong denoise): conv_gamma=10.0, lin_gamma=1.0
    # ------------------------------------------------------------------
    CONV_GAMMA = 1.0   # Gamma for Conv2d layers (patch embedding)
    LIN_GAMMA = 0.5    # Gamma for Linear layers (attention & MLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Gamma rule: Conv2d gamma={CONV_GAMMA}, Linear gamma={LIN_GAMMA}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_dataset = CUBDataset(
        root=str(data_root),
        split="test",
        transform=None,  # keep CUBDataset's internal ViT transforms
    )

    print(f"Dataset size: {len(test_dataset)}")
    print(f"Will generate LRP for {num_samples} samples")

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    model = create_vit_cub_model(
        num_classes=200,
        pretrained=False,
        freeze_backbone=False,
    )

    ckpt = torch.load("checkpoints/vit_cub_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Pick sample indices (random, reproducible)
    # Note: Sample selection is deterministic (seed=42) and independent of gamma params
    # ------------------------------------------------------------------
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), k=min(num_samples, len(test_dataset)))

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample_idx in enumerate(sample_indices, start=1):
        print(f"\n[{i}/{len(sample_indices)}] Processing sample_idx={sample_idx} ...")
        try:
            # Compute LRP with Gamma rule enhancement
            lrp_result = compute_lrp_for_sample(
                model, test_dataset, sample_idx, device,
                conv_gamma=CONV_GAMMA,
                lin_gamma=LIN_GAMMA,
            )
            meta = save_visualizations(lrp_result, sample_idx, output_dir)

            # Add gamma parameters to metadata for traceability
            meta["gamma_params"] = lrp_result["gamma_params"]

            results.append(meta)

            print(f"  ✓ Target class: {meta['target_class']} (label: {meta['label']})")
            print(f"  ✓ Saved: {meta['files']['overlay']}")
            print(f"  ✓ Saved: {meta['files']['comparison']}")

        except Exception as e:
            print(f"  ✗ Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ------------------------------------------------------------------
    # Save paths + JSON metadata (includes gamma parameters)
    # ------------------------------------------------------------------
    paths_file = output_dir / "lrp_image_paths.txt"
    with open(paths_file, "w") as f:
        for r in results:
            f.write(f"{r['sample_idx']:06d}: {r['img_path']}\n")

    json_file = output_dir / "lrp_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Completed! Generated {len(results)}/{len(sample_indices)} samples")
    print(f"Gamma parameters: Conv2d={CONV_GAMMA}, Linear={LIN_GAMMA}")
    print(f"Output directory: {output_dir}")
    print(f"Image paths saved to: {paths_file}")
    print(f"Results JSON saved to: {json_file}")


if __name__ == "__main__":
    main()
