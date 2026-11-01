"""
Temporal Video Deepfake Classifier Module
Aggregates frame features across time using Bidirectional LSTM and
Temporal Attention Pooling for video-level authenticity classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.models.cnn import FaceCNNFeatureExtractor, LightweightFaceCNN


class TemporalAttention(nn.Module):
    """
    Computes attention scores over video frames to focus on temporally inconsistent frames.
    """

    def __init__(self, input_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, input_dim)
        Returns:
            weighted_context: (batch_size, input_dim)
            weights: (batch_size, seq_len)
        """
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, input_dim)
        return context, weights.squeeze(-1)


class VideoDeepfakeModel(nn.Module):
    """
    End-to-end or feature-level Video Deepfake Classifier.
    Combines frame-level feature extractor with temporal BiLSTM and attention pooling.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 2,
        use_lightweight: bool = False
    ):
        super(VideoDeepfakeModel, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        if use_lightweight:
            self.cnn = LightweightFaceCNN(feature_dim=feature_dim)
        else:
            self.cnn = FaceCNNFeatureExtractor(pretrained=False, feature_dim=feature_dim)

        # Temporal Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = hidden_dim * 2
        self.attention = TemporalAttention(lstm_out_dim)

        # Frame-level classifier head
        self.frame_head = nn.Linear(feature_dim, num_classes)

        # Video-level classifier head
        self.video_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward_frames(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        frames: (batch_size, seq_len, 3, H, W)
        """
        b, seq_len, c, h, w = frames.shape
        flat_frames = frames.view(b * seq_len, c, h, w)

        frame_feats, _ = self.cnn(flat_frames)  # (b * seq_len, feature_dim)
        frame_feats = frame_feats.view(b, seq_len, self.feature_dim)

        # Frame-level classification logits
        frame_logits = self.frame_head(frame_feats)  # (b, seq_len, 2)

        # Temporal modeling
        lstm_out, _ = self.lstm(frame_feats)  # (b, seq_len, hidden_dim * 2)
        video_feat, attn_weights = self.attention(lstm_out)  # (b, hidden_dim * 2)

        # Video-level classification logits
        video_logits = self.video_head(video_feat)  # (b, 2)

        return {
            "video_logits": video_logits,
            "frame_logits": frame_logits,
            "video_features": video_feat,
            "attention_weights": attn_weights
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward returning video-level logits.
        """
        res = self.forward_frames(x)
        return res["video_logits"]
