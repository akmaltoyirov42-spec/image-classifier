import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_loaders
from src.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = model(images)
        total_loss += criterion(out, labels).item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(data_dir: str = "data", epochs_head: int = 5, epochs_finetune: int = 10, batch_size: int = 32):
    train_loader, val_loader, classes = get_loaders(data_dir, batch_size)
    print(f"Classes: {classes}  |  Device: {DEVICE}")

    model = build_model(len(classes), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # phase 1 — train head only, backbone frozen
    print("\nPhase 1: head only")
    optimizer = Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)
    for epoch in range(epochs_head):
        _, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        _, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        print(f"  {epoch+1}/{epochs_head}  train={tr_acc:.3f}  val={val_acc:.3f}")

    # phase 2 — unfreeze everything, lower lr
    print("\nPhase 2: full fine-tune")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs_finetune):
        _, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        _, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        print(f"  {epoch+1}/{epochs_finetune}  train={tr_acc:.3f}  val={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pth")
    (out_dir / "classes.json").write_text(json.dumps(classes))
    print(f"\nBest val acc: {best_acc:.4f} — saved to {out_dir}/")


if __name__ == "__main__":
    train()
