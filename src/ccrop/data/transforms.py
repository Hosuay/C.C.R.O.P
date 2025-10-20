"""
Image transformations for training and inference
"""

import torchvision.transforms as transforms
from typing import Tuple


def get_train_transforms(input_size: int = 224, augment: bool = True) -> transforms.Compose:
    """
    Get training transforms with optional augmentation

    Args:
        input_size: Input image size
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation)

    Args:
        input_size: Input image size

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_inference_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms

    Args:
        input_size: Input image size

    Returns:
        Composed transforms
    """
    return get_val_transforms(input_size)


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image

    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    import torch
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean
