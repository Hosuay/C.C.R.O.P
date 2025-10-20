"""
Disease classification model using ResNet/EfficientNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, List
from pathlib import Path


class CCROPClassifier(nn.Module):
    """
    Cannabis Leaf Disease Classifier using transfer learning

    Supports multiple architectures:
    - ResNet18, ResNet34, ResNet50
    - EfficientNet-B0, B1, B2
    """

    def __init__(self, arch: str = "resnet18", num_classes: int = 38,
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the classifier

        Args:
            arch: Model architecture (resnet18, resnet34, resnet50, efficientnet_b0, etc.)
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers (only train classifier head)
        """
        super(CCROPClassifier, self).__init__()

        self.arch = arch
        self.num_classes = num_classes

        # Create backbone
        if arch == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif arch == "resnet34":
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif arch == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported architecture: {arch}. "
                           f"Choose from: resnet18, resnet34, resnet50, "
                           f"efficientnet_b0, efficientnet_b1, efficientnet_b2")

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone layers except the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded model and metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model config
        config = checkpoint.get('config', {})
        arch = checkpoint.get('architecture', config.get('MODEL_ARCH', 'resnet18'))
        classes = checkpoint['classes']
        num_classes = len(classes)

        # Create model
        model = cls(arch=arch, num_classes=num_classes, pretrained=False)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        metadata = {
            'classes': classes,
            'stress_mapping': checkpoint.get('stress_mapping', {}),
            'architecture': arch,
            'val_loss': checkpoint.get('val_loss'),
            'val_acc': checkpoint.get('val_acc'),
            'epoch': checkpoint.get('epoch')
        }

        return model, metadata

    def save_checkpoint(self, save_path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint

        Args:
            save_path: Path to save checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': self.arch,
            'num_classes': self.num_classes,
        }

        if metadata:
            checkpoint.update(metadata)

        torch.save(checkpoint, save_path)
        print(f"Model checkpoint saved to {save_path}")


def create_classifier(arch: str = "resnet18", num_classes: int = 38,
                     pretrained: bool = True) -> CCROPClassifier:
    """
    Factory function to create a classifier

    Args:
        arch: Model architecture
        num_classes: Number of classes
        pretrained: Use pretrained weights

    Returns:
        CCROPClassifier instance
    """
    return CCROPClassifier(arch=arch, num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test classifier creation
    print("Testing CCROPClassifier...")

    model = create_classifier(arch="resnet18", num_classes=38)
    print(f"\nModel created: {model.arch}")
    print(f"Parameters: {model.count_parameters()}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    print("\nClassifier test passed!")
