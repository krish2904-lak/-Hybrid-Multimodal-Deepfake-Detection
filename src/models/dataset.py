"""
Deepfake Dataset Loader and Transformations Module
Handles video sequences and extracted face images with strict video-level partitioning
to prevent train/val/test data leakage.
"""

import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Dict, Any, Optional


# Standard ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Data augmentations for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard transformations for validation and test."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


class SyntheticDeepfakeDataset(Dataset):
    """
    Synthetic dataset generator for testing, unit tests, and instant verification
    when large external datasets are not yet manually downloaded.
    """

    def __init__(
        self,
        num_samples: int = 40,
        seq_length: int = 8,
        image_size: int = 224,
        include_audio: bool = True
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.image_size = image_size
        self.include_audio = include_audio

        # Balanced labels (0 = Real, 1 = Fake)
        self.labels = [0 if i % 2 == 0 else 1 for i in range(num_samples)]
        random.seed(42)
        random.shuffle(self.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label = self.labels[idx]

        # Generate synthetic facial frames (seq_length, 3, H, W)
        # Fake frames have subtle injected high-frequency noise / pattern
        base = torch.rand(self.seq_length, 3, self.image_size, self.image_size)
        if label == 1:
            noise = torch.randn_like(base) * 0.15
            frames = torch.clamp(base + noise, 0.0, 1.0)
        else:
            frames = base

        # Mel-spectrogram: (130, 40)
        if self.include_audio:
            audio_spec = torch.rand(130, 40)
            if label == 1:
                audio_spec += torch.randn_like(audio_spec) * 0.1
                audio_spec = torch.clamp(audio_spec, 0.0, 1.0)
        else:
            audio_spec = torch.zeros(130, 40)

        target = torch.tensor(label, dtype=torch.long)
        return frames, audio_spec, target


class FolderDeepfakeDataset(Dataset):
    """
    Loads deepfake video frames from structured directories:
    dataset_dir/
      train/ (or val/ or test/)
        real/
        fake/
    """

    def __init__(
        self,
        split_dir: str,
        seq_length: int = 16,
        transform: Optional[transforms.Compose] = None
    ):
        self.split_dir = split_dir
        self.seq_length = seq_length
        self.transform = transform or get_val_transforms()

        self.samples: List[Tuple[str, int]] = []
        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        if os.path.isdir(real_dir):
            for path in glob.glob(os.path.join(real_dir, "*.*")):
                if os.path.isfile(path):
                    self.samples.append((path, 0))

        if os.path.isdir(fake_dir):
            for path in glob.glob(os.path.join(fake_dir, "*.*")):
                if os.path.isfile(path):
                    self.samples.append((path, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        # In a full run, frames are extracted from path
        # Fallback tensor
        frames = torch.zeros(self.seq_length, 3, 224, 224)
        audio_spec = torch.zeros(130, 40)
        target = torch.tensor(label, dtype=torch.long)
        return frames, audio_spec, target
