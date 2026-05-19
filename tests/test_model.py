import torch
from src.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=10)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (4, 10)


def test_frozen_backbone_has_no_grad():
    model = build_model(num_classes=5, freeze_backbone=True)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_unfrozen_model_all_grad():
    model = build_model(num_classes=5, freeze_backbone=False)
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} should have grad"


def test_different_class_counts():
    for n in [2, 5, 100]:
        model = build_model(num_classes=n)
        out = model(torch.randn(1, 3, 224, 224))
        assert out.shape[1] == n
