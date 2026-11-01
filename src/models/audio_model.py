"""
Audio Deepfake Detection Model
Processes Mel-Spectrogram sequences using LSTM and dense representations
for acoustic artifact detection in multimodal deepfake classification.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class AudioDeepfakeModel(nn.Module):
    """
    Processes Mel-Spectrogram sequences (time_steps, n_mels)
    using LSTM temporal encoder and dense classification head.
    """

    def __init__(
        self,
        n_mels: int = 40,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_classes: int = 2
    ):
        super(AudioDeepfakeModel, self).__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Temporal LSTM encoder over spectrogram time frames
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Dense projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU()
        )

        # Standalone audio classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )

    def extract_embedding(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        spectrogram: (batch_size, time_steps, n_mels)
        Returns: (batch_size, embedding_dim)
        """
        # LSTM processes sequence
        lstm_out, (hn, cn) = self.lstm(spectrogram)
        # Use last hidden state or last output step
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        embedding = self.projection(last_step)  # (batch_size, embedding_dim)
        return embedding

    def forward(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (audio_embedding, logits)
        """
        embedding = self.extract_embedding(spectrogram)
        logits = self.classifier(embedding)
        return embedding, logits
