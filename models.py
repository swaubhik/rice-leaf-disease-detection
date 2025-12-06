"""Model architectures for rice leaf disease classification."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights
)
from typing import Optional


class RiceLeafClassifier(nn.Module):
    """Rice Leaf Disease Classifier using transfer learning."""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """Initialize the classifier.
        
        Args:
            num_classes: Number of disease classes
            model_name: Name of the backbone model ('resnet50', 'resnet101', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for the classifier head
        """
        super(RiceLeafClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone model
        if model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original FC layer
            
        elif model_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone parameters for feature extraction."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    num_classes: int,
    model_name: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.5,
    device: Optional[str] = None
) -> RiceLeafClassifier:
    """Create a rice leaf classifier model.
    
    Args:
        num_classes: Number of disease classes
        model_name: Name of the backbone model
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        device: Device to move model to ('cuda' or 'cpu')
        
    Returns:
        Initialized model
    """
    model = RiceLeafClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout
    )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test model creation
    num_classes = 5
    model = create_model(num_classes=num_classes, model_name="resnet50")
    
    print(f"Model: {model.model_name}")
    print(f"Number of classes: {model.num_classes}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
