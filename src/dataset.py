from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet mean/std — standard for any pretrained torchvision model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(root / "val", transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes
