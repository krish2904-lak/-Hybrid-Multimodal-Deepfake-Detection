"""
CNN Visual Feature Extractor Module
Supports lightweight convolutional backbone and Transfer Learning (MobileNetV2 / ResNet)
for face-level spatial artifact detection.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class FaceCNNFeatureExtractor(nn.Module):
    """
    Extracts deep visual embeddings from face crops.
    Default backbone: MobileNetV2 for high efficiency and mobile/CPU deployment.
    """

    def __init__(self, pretrained: bool = False, feature_dim: int = 512):
        super(FaceCNNFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim

        # Backbone: MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Project backbone output (1280) to feature_dim
        self.projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Standalone frame classifier head (Binary: 0=Real, 1=Fake)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 2)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 3, H, W)
        Returns: (batch_size, feature_dim)
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        projected = self.projection(flattened)
        return projected

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 3, H, W)
        Returns: (features, logits)
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return feats, logits


class LightweightFaceCNN(nn.Module):
    """
    Fast custom CNN for edge inference, simulation, and test validation.
    """

    def __init__(self, feature_dim: int = 256):
        super(LightweightFaceCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return feats, logits
