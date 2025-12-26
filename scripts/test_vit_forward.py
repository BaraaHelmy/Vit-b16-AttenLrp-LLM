# scripts/test_vit_forward.py
#
# Purpose:
# - Sanity-check ViT forward pass on CUB
# - Enable AttnLRP/CP-LRP via LXT monkey-patching for torchvision ViT
# - Compute pixel-level relevance for MULTIPLE images via backward()
# - Save for each sample:
#     1) lrp_heatmap_XXX.png         (heatmap only)
#     2) lrp_overlay_XXX.png         (original image + heatmap overlay)
#     3) lrp_relevance_raw_XXX.pt    (raw relevance tensor)
#     4) lrp_image_paths.txt         (all image paths)
#     5) lrp_results.json            (metadata JSON)

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from cub_dataset import CUBDataset
from vit_cub_model import create_vit_cub_model

# ------------------------------------------------------------------
# LXT (LRP-eXplains-Transformers) for torchvision ViT in THIS repo:
# It works via monkey-patching torchvision.models.vision_transformer.
# This MUST happen before creating the ViT model.
# ------------------------------------------------------------------
from torchvision.models import vision_transformer
from lxt.efficient import monkey_patch

monkey_patch(vision_transformer, verbose=True)  # Enables LRP rules in backward pass


def compute_lrp_for_sample(model, dataset, sample_idx, device):
    """
    Compute LRP relevance for a single sample.
    
    Args:
        model: The ViT model (already on device and in eval mode)
        dataset: CUBDataset instance
        sample_idx: Index of the sample to analyze
        device: torch device
    
    Returns:
        dict with 'img_path', 'target_class', 'relevance', 'x'
    """
    # Get image path and tensor
    img_path, label = dataset.samples[sample_idx]
    x_tensor, _ = dataset[sample_idx]
    x = x_tensor.unsqueeze(0).to(device).requires_grad_(True)  # [1, 3, 224, 224]
    
    # Clear gradients
    model.zero_grad(set_to_none=True)
    
    # Forward pass
    logits = model(x)  # [1, 200]
    target_class = logits.argmax(dim=1).item()
    
    # Backward relevance
    logits[0, target_class].backward()
    
    # Get relevance
    relevance = x.grad  # [1, 3, 224, 224]
    
    return {
        'img_path': str(img_path),
        'target_class': target_class,
        'relevance': relevance.detach().cpu(),
        'x': x.detach().cpu(),
    }


def save_visualizations(lrp_result, sample_idx, output_dir):
    """
    Save heatmap and overlay visualizations for a sample.
    
    Args:
        lrp_result: dict from compute_lrp_for_sample()
        sample_idx: Index of the sample (for filename)
        output_dir: Path to output directory
    """
    import matplotlib.pyplot as plt
    
    relevance = lrp_result['relevance']
    x = lrp_result['x']
    
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: 000, 001, 002, etc.
    idx_str = f"{sample_idx:03d}"
    
    # 1) Heatmap-only PNG
    heat = relevance[0].sum(dim=0)  # [224, 224]
    heat = heat / (heat.abs().max() + 1e-8)  # normalize to [-1, 1]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(heat, cmap="bwr", vmin=-1, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"lrp_heatmap_{idx_str}.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # 2) Overlay PNG
    img_disp = x[0]
    img_disp = (img_disp * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    img_disp = img_disp.permute(1, 2, 0).numpy()   # HWC
    
    heat_np = heat.numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_disp)
    plt.imshow(heat_np, cmap="bwr", vmin=-1, vmax=1, alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"lrp_overlay_{idx_str}.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # 3) Save raw relevance tensor
    torch.save(relevance, output_dir / f"lrp_relevance_raw_{idx_str}.pt")


def main():
    data_root = Path("dataset/raw/CUB_200_2011")
    output_dir = Path("results/lrp_samples")
    num_samples = 20  # Number of samples to generate
    
    batch_size = 32
    num_workers = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    train_dataset = CUBDataset(
        root=str(data_root),
        split="train",
        transform=None,  # keep CUBDataset's internal ViT transforms
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Will generate LRP for {num_samples} samples")
    
    # ------------------------------------------------------------------
    # Model creation + checkpoint loading
    # ------------------------------------------------------------------
    model = create_vit_cub_model(
        num_classes=200,
        pretrained=False,
        freeze_backbone=False,
    )
    
    ckpt = torch.load(
        "checkpoints/vit_cub_best.pt",
        map_location="cpu",
    )
    
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    
    # ------------------------------------------------------------------
    # Generate LRP for multiple samples
    # ------------------------------------------------------------------
    import random
    random.seed(42)  # For reproducibility
    
    # Select random sample indices (or use sequential: range(num_samples))
    sample_indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
    
    results = []
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{num_samples}] Processing sample {sample_idx}...")
        
        try:
            # Compute LRP
            lrp_result = compute_lrp_for_sample(model, train_dataset, sample_idx, device)
            
            # Save visualizations and raw data
            save_visualizations(lrp_result, sample_idx, output_dir)
            
            # Store result info
            results.append({
                'sample_idx': sample_idx,
                'img_path': lrp_result['img_path'],
                'target_class': lrp_result['target_class'],
            })
            
            print(f"  ✓ Saved: lrp_overlay_{sample_idx:03d}.png")
            print(f"  ✓ Target class: {lrp_result['target_class']}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all image paths to a file
    paths_file = output_dir / "lrp_image_paths.txt"
    with open(paths_file, "w") as f:
        for result in results:
            f.write(f"{result['sample_idx']:03d}: {result['img_path']}\n")
    
    # Also save as JSON for easier parsing
    json_file = output_dir / "lrp_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Completed! Generated {len(results)}/{num_samples} samples")
    print(f"Output directory: {output_dir}")
    print(f"Image paths saved to: {paths_file}")
    print(f"Results JSON saved to: {json_file}")


if __name__ == "__main__":
    main()
