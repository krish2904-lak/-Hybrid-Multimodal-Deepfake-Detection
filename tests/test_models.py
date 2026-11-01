"""
Unit tests for Deep Learning Models:
- Visual CNN Feature Extractor
- Temporal Video BiLSTM Model
- Acoustic Spectrogram LSTM Model
- Hybrid Multimodal Cross-Modal Fusion
"""

import torch
import pytest

from src.models.cnn import LightweightFaceCNN, FaceCNNFeatureExtractor
from src.models.video_model import VideoDeepfakeModel
from src.models.audio_model import AudioDeepfakeModel
from src.models.fusion_model import HybridMultimodalFusionModel


def test_lightweight_cnn():
    model = LightweightFaceCNN(feature_dim=128)
    x = torch.rand(2, 3, 112, 112)
    feats, logits = model(x)
    assert feats.shape == (2, 128)
    assert logits.shape == (2, 2)


def test_video_bilstm_model():
    model = VideoDeepfakeModel(feature_dim=128, hidden_dim=64, use_lightweight=True)
    frames = torch.rand(2, 6, 3, 112, 112)  # (batch=2, seq_len=6, C=3, H=112, W=112)
    res = model.forward_frames(frames)

    assert res["video_logits"].shape == (2, 2)
    assert res["frame_logits"].shape == (2, 6, 2)
    assert res["video_features"].shape == (2, 128)  # hidden_dim * 2
    assert res["attention_weights"].shape == (2, 6)


def test_audio_lstm_model():
    model = AudioDeepfakeModel(n_mels=40, hidden_dim=64, embedding_dim=64)
    spec = torch.rand(2, 130, 40)  # (batch=2, time_steps=130, n_mels=40)
    embedding, logits = model(spec)

    assert embedding.shape == (2, 64)
    assert logits.shape == (2, 2)


def test_hybrid_multimodal_fusion():
    model = HybridMultimodalFusionModel(
        video_feat_dim=256,
        audio_feat_dim=64,
        fusion_dim=128,
        num_classes=2,
        use_lightweight_video=True
    )
    frames = torch.rand(2, 4, 3, 112, 112)
    spec = torch.rand(2, 130, 40)

    # 1. Multimodal forward
    out_multi = model(frames, spec, has_audio=True)
    assert out_multi["logits"].shape == (2, 2)
    assert out_multi["probabilities"].shape == (2, 2)
    assert out_multi["modality_used"] == "multimodal_fusion"

    # 2. Unimodal visual fallback (no audio)
    out_visual = model(frames, None, has_audio=False)
    assert out_visual["logits"].shape == (2, 2)
    assert out_visual["probabilities"].shape == (2, 2)
    assert out_visual["modality_used"] == "visual_only"
