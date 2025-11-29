#!/usr/bin/env python3
"""
Simple CNN for binary classification (pothole vs background).

Architecture options:
1. SimpleCNN: Lightweight custom CNN
2. ResNet18Classifier: Transfer learning with pretrained ResNet18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network for binary classification.
    
    Architecture:
        - 3 conv blocks (conv -> bn -> relu -> maxpool)
        - 2 fully connected layers
        - Binary classification output
    
    Args:
        num_classes: Number of output classes (default 2 for binary)
        input_channels: Number of input channels (3 for RGB)
        dropout: Dropout probability
    """
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # For 64x64 input: after 3 poolings -> 8x8 spatial size
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 64x64 -> 32x32
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNet18Classifier(nn.Module):
    """
    Transfer learning classifier using pretrained ResNet18.
    
    Replaces the final FC layer for binary classification.
    
    Args:
        num_classes: Number of output classes (default 2)
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: If True, freeze all layers except final FC
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)


def get_model(model_type: str = 'simple', num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'simple' or 'resnet18'
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        PyTorch model
    """
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'resnet18':
        return ResNet18Classifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    print("Testing SimpleCNN...")
    model = SimpleCNN(num_classes=2)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting ResNet18Classifier...")
    model = ResNet18Classifier(num_classes=2, pretrained=False)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
