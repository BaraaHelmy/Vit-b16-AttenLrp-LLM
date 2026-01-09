# Vit-b16-AttenLrp-LLM

Vision Transformer (ViT-B/16) with Attention-aware Layer-wise Relevance Propagation (AttnLRP) for explainable bird classification on the CUB-200-2011 dataset.

## Overview

This project implements:
- Fine-tuned ViT-B/16 on CUB-200-2011 (200 bird species classification)
- AttnLRP via LXT (LRP-eXplains-Transformers) for model interpretability
- **Gamma rule enhancement** for improved heatmap quality (noise reduction, positive relevance emphasis)
- Visualization tools for LRP heatmaps and overlays

## Project Structure

```
thesis/
├── scripts/
│   ├── test_vit_forward.py      # LRP heatmap generation with Gamma rule
│   └── inspect_lrp_predictions.py  # Inspect predictions for LRP samples
├── checkpoints/
│   └── vit_cub_best.pt          # Best model checkpoint
├── dataset/
│   └── raw/CUB_200_2011/        # CUB-200-2011 dataset
├── results/
│   └── lrp_samples/             # Generated LRP visualizations
├── external/
│   └── LXT/                     # LRP-eXplains-Transformers library
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd thesis

# Install dependencies
pip install torch torchvision
pip install lxt zennit
pip install matplotlib pillow numpy
```

## LRP with Gamma Rule Enhancement

### Why Gamma Rule?

Vision Transformers are susceptible to **gradient shattering**, which leads to very noisy heatmaps. The Gamma rule is an LRP propagation rule that:

1. **Reduces noise**: Smooths out scattered artifacts in heatmaps
2. **Emphasizes positive relevance**: Highlights features that positively contribute to the prediction
3. **Improves signal-to-noise ratio**: Produces cleaner, more interpretable visualizations

### How It Works

The Gamma rule modifies the standard LRP relevance propagation:

```
Standard LRP:  R_j = sum_k (a_j * w_jk / sum_i a_i * w_ik) * R_k

Gamma LRP:     R_j = sum_k (a_j * (w_jk + γ * max(0, w_jk)) / 
                           (sum_i a_i * (w_ik + γ * max(0, w_ik)))) * R_k
```

Where `γ` (gamma) controls how much positive contributions are emphasized:
- **γ = 0**: Standard LRP (no gamma enhancement)
- **Higher γ**: More positive emphasis, less noise

**Important**: The Gamma rule only affects the backward pass (LRP computation). The forward pass (model predictions, confidence scores) remains **unchanged**.

### Parameter Tuning Guide

| Configuration | Conv2d γ | Linear γ | Effect |
|--------------|----------|----------|--------|
| Conservative | 0.5 | 0.1 | Moderate noise reduction, balanced relevance |
| **Balanced (Recommended)** | **1.0** | **0.5** | Good noise reduction, clear positive emphasis |
| Aggressive | 10.0 | 1.0 | Strong noise reduction, strong positive emphasis |
| Very Aggressive | 100.0 | 1.0 | Maximum positive emphasis, minimal noise |

**Recommended starting values**: `conv_gamma=1.0`, `lin_gamma=0.5`

### Adjusting Gamma Parameters

Edit `scripts/test_vit_forward.py` and modify the configuration section:

```python
# Gamma Rule Configuration
CONV_GAMMA = 1.0   # Gamma for Conv2d layers (patch embedding)
LIN_GAMMA = 0.5    # Gamma for Linear layers (attention & MLP)
```

### Visual Comparison

| Without Gamma (Baseline) | With Gamma (1.0, 0.5) |
|--------------------------|----------------------|
| Noisy, scattered relevance | Clean, focused heatmaps |
| Many small positive/negative spots | Stronger positive regions |
| Harder to identify key features | Clear identification of important areas |

## Usage

### Generate LRP Heatmaps

```bash
cd thesis
python scripts/test_vit_forward.py
```

**Expected Output:**
```
Using device: cuda
Using Gamma rule: Conv2d gamma=1.0, Linear gamma=0.5
Dataset size: 5794
Will generate LRP for 20 samples
Model loaded successfully.

[1/20] Processing sample_idx=1234 ...
  ✓ Target class: 45 (label: 42)
  ✓ Saved: lrp_overlay_001234.png
  ✓ Saved: lrp_comparison_001234.png
...

============================================================
Completed! Generated 20/20 samples
Gamma parameters: Conv2d=1.0, Linear=0.5
Output directory: results/lrp_samples
```

### Output Files

For each sample, the script generates:
- `lrp_heatmap_XXXXXX.png` - Heatmap only
- `lrp_overlay_XXXXXX.png` - Image with heatmap overlay
- `lrp_comparison_XXXXXX.png` - 3-panel comparison
- `lrp_relevance_raw_XXXXXX.pt` - Raw relevance tensor

Plus metadata files:
- `lrp_image_paths.txt` - List of processed image paths
- `lrp_results.json` - Full metadata including gamma parameters

### JSON Output Format

```json
[
  {
    "sample_idx": 1234,
    "img_path": "/path/to/image.jpg",
    "label": 42,
    "target_class": 45,
    "gamma_params": {
      "conv_gamma": 1.0,
      "lin_gamma": 0.5
    },
    "files": {
      "heatmap": "lrp_heatmap_001234.png",
      "overlay": "lrp_overlay_001234.png",
      "comparison": "lrp_comparison_001234.png",
      "relevance_raw": "lrp_relevance_raw_001234.pt"
    }
  }
]
```

### Inspect Predictions

```bash
python scripts/inspect_lrp_predictions.py
```

This outputs the confidence scores and predictions for the generated LRP samples.

## Technical Details

### Dependencies

- `torch`, `torchvision` - PyTorch and Vision Transformer model
- `lxt` - LRP-eXplains-Transformers library for AttnLRP
- `zennit` - Provides Gamma rule implementation
- `matplotlib` - Visualization
- `PIL`, `numpy` - Image processing

### How LRP + Gamma Works in This Implementation

1. **Monkey-patch torchvision**: LXT modifies `vision_transformer` module to enable LRP in backward pass
2. **Monkey-patch zennit**: Makes zennit's explicit LRP compatible with LXT's Input*Gradient formulation
3. **Register Gamma composite**: Applies Gamma rule to Conv2d and Linear layers via `LayerMapComposite`
4. **Forward pass**: Normal model inference (unchanged)
5. **Backward pass**: LRP with Gamma rule computes pixel-level relevance
6. **Cleanup**: Remove composite after each sample to prevent interference

### Sample Selection

Sample indices are selected with `random.seed(42)` for reproducibility. The same 20 samples are processed regardless of gamma parameters, allowing fair comparison of heatmap quality across different configurations.

## References

- [LRP-eXplains-Transformers (LXT)](https://github.com/rachtibat/LRP-for-Transformers)
- [zennit: A PyTorch library for LRP](https://github.com/chr5tphr/zennit)
- [AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers (ICML 2024)](https://arxiv.org/abs/2402.05602)
- [CUB-200-2011 Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

## License

See LICENSE file for details.
