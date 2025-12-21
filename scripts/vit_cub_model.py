# scripts/vit_cub_model.py

from typing import Tuple

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def create_vit_cub_model(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ViT-B/16 model adapted for CUB-200.

    - Loads ImageNet-1k pretrained weights (optional)
    - Replaces the classifier head with a new Linear(768 -> num_classes)
    - Optionally freezes all backbone parameters (feature extractor mode)
    """

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)

    # Optionally freeze all existing parameters (backbone)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # The classifier head lives in model.heads (a Sequential with submodule "head")
    # e.g.: Sequential((head): Linear(in_features=768, out_features=1000, bias=True))
    in_features = model.heads.head.in_features  # 768 for ViT-B/16

    model.heads.head = nn.Linear(in_features, num_classes)

    return model
