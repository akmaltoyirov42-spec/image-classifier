import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    EfficientNet-B0 with a replaced classifier head.
    Frozen backbone = only the head trains on first pass (much faster, less overfitting on small datasets).
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
