"""
Hybrid Multimodal Fusion Architecture for Deepfake Detection
Combines visual features from video branch and acoustic features from audio branch
into an integrated cross-modal representation with graceful unimodal fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.models.video_model import VideoDeepfakeModel
from src.models.audio_model import AudioDeepfakeModel


class HybridMultimodalFusionModel(nn.Module):
    """
    Hybrid Multimodal Architecture:
    Video Branch (BiLSTM Attention) + Audio Branch (Mel LSTM) -> Cross-Modal Fusion -> Classifier
    """

    def __init__(
        self,
        video_feat_dim: int = 256,  # BiLSTM output (hidden_dim * 2 = 128 * 2)
        audio_feat_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 2,
        use_lightweight_video: bool = False
    ):
        super(HybridMultimodalFusionModel, self).__init__()
        self.video_feat_dim = video_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        # Sub-networks
        self.video_branch = VideoDeepfakeModel(
            feature_dim=256,
            hidden_dim=128,
            num_classes=num_classes,
            use_lightweight=use_lightweight_video
        )
        self.audio_branch = AudioDeepfakeModel(
            n_mels=40,
            hidden_dim=64,
            embedding_dim=audio_feat_dim,
            num_classes=num_classes
        )

        # Cross-modal fusion layer
        combined_dim = video_feat_dim + audio_feat_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # Unimodal video fallback head (when audio is missing)
        self.unimodal_video_head = nn.Sequential(
            nn.Linear(video_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward_features(
        self,
        video_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fast forward given pre-extracted visual and acoustic feature vectors.
        video_features: (batch_size, video_feat_dim)
        audio_features: (batch_size, audio_feat_dim) or None
        """
        batch_size = video_features.size(0)

        if audio_features is None or (audio_mask is not None and not audio_mask.any()):
            # Fallback to unimodal visual head
            return self.unimodal_video_head(video_features)

        # Concatenate features
        fused_input = torch.cat([video_features, audio_features], dim=1)
        return self.fusion_net(fused_input)

    def forward(
        self,
        frames: torch.Tensor,
        spectrogram: Optional[torch.Tensor] = None,
        has_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Full end-to-end forward pass from raw frames and spectrogram.
        frames: (batch_size, seq_len, 3, H, W)
        spectrogram: (batch_size, time_steps, n_mels)
        """
        # Video branch forward
        video_res = self.video_branch.forward_frames(frames)
        video_feats = video_res["video_features"]  # (b, video_feat_dim)
        frame_logits = video_res["frame_logits"]  # (b, seq_len, 2)
        attn_weights = video_res["attention_weights"]  # (b, seq_len)

        if has_audio and spectrogram is not None:
            audio_feats, audio_logits = self.audio_branch(spectrogram)
            fused_input = torch.cat([video_feats, audio_feats], dim=1)
            logits = self.fusion_net(fused_input)
            modality_used = "multimodal_fusion"
        else:
            audio_feats = None
            audio_logits = None
            logits = self.unimodal_video_head(video_feats)
            modality_used = "visual_only"

        probs = F.softmax(logits, dim=-1)
        frame_probs = F.softmax(frame_logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "frame_probabilities": frame_probs,
            "video_features": video_feats,
            "audio_features": audio_feats,
            "audio_logits": audio_logits,
            "attention_weights": attn_weights,
            "modality_used": modality_used
        }
