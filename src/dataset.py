"""Dataset loading and preprocessing for Fashion-MNIST and CIFAR-10."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from src.config import DataConfig


DATASET_REGISTRY: dict[str, type[datasets.VisionDataset]] = {
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def _build_transform(dataset_name: str) -> transforms.Compose:
    """Build a preprocessing transform that normalizes images to [-1, 1].

    Args:
        dataset_name (str): One of "fashion_mnist" or "cifar10".

    Returns:
        transforms.Compose: Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # scales to [0, 1]
        transforms.Normalize(
            mean=[0.5] * (1 if dataset_name == "fashion_mnist" else 3),
            std=[0.5] * (1 if dataset_name == "fashion_mnist" else 3),
        ),
    ])


def create_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        config (DataConfig): Data configuration.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader).
    """
    dataset_cls = DATASET_REGISTRY.get(config.dataset)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    transform = _build_transform(config.dataset)

    full_train = dataset_cls(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    val_size = int(len(full_train) * config.val_split)
    train_size = len(full_train) - val_size

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
