"""
End-to-End Inference Predictor for Deepfake Detection
Executes complete pipeline:
Video Validation -> Frame Extraction -> Face Detection -> Audio Extraction ->
Model Forward -> Video & Frame Predictions -> Downloadable Report Diagnostics.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, List

from src.preprocessing.extract_frames import validate_video_file, VideoFrameExtractor
from src.preprocessing.detect_faces import FaceDetector
from src.preprocessing.audio_extract import AudioExtractor
from src.models.fusion_model import HybridMultimodalFusionModel


class DeepfakePredictor:
    """
    Unified Inference Engine for Deepfake Detection.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_model.pt",
        device: Optional[torch.device] = None,
        sequence_length: int = 16,
        target_size: tuple = (112, 112)
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.target_size = target_size

        # Preprocessing sub-modules
        self.frame_extractor = VideoFrameExtractor(target_size=target_size)
        self.face_detector = FaceDetector(target_size=target_size, padding_ratio=0.25)
        self.audio_extractor = AudioExtractor()

        # Model instantiation
        self.model = HybridMultimodalFusionModel(
            video_feat_dim=256,
            audio_feat_dim=64,
            fusion_dim=128,
            num_classes=2,
            use_lightweight_video=True
        ).to(self.device)

        self.model_loaded = False
        self.checkpoint_path = checkpoint_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"[Predictor] Successfully loaded model checkpoint from: {checkpoint_path}")
            except Exception as e:
                print(f"[Predictor] Warning: Failed loading checkpoint {checkpoint_path}: {e}")

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a video file.
        Returns comprehensive analysis dictionary.
        """
        start_time = time.time()

        # Step 1: Validate Video
        is_valid, val_msg = validate_video_file(video_path)
        if not is_valid:
            return {
                "success": False,
                "error": val_msg,
                "video_path": video_path,
                "processing_time": round(time.time() - start_time, 2)
            }

        # Step 2: Extract Frames
        try:
            extract_res = self.frame_extractor.extract_sequence(
                video_path,
                sequence_length=self.sequence_length
            )
            raw_frames = extract_res["frames"]
            metadata = extract_res["metadata"]
        except Exception as e:
            return {
                "success": False,
                "error": f"Frame extraction failed: {str(e)}",
                "video_path": video_path,
                "processing_time": round(time.time() - start_time, 2)
            }

        # Step 3: Detect and Crop Faces
        face_res = self.face_detector.process_sequence(raw_frames)
        face_crops = face_res["face_crops"]  # (seq_len, H, W, 3)
        faces_detected = face_res["detected_count"]

        # Step 4: Extract Audio Spectrogram
        audio_res = self.audio_extractor.extract_spectrogram(video_path)
        audio_spec = audio_res["spectrogram"]
        has_audio = audio_res["has_audio"]

        # Step 5: PyTorch Tensor Formatting
        # Face crops: (seq_len, H, W, 3) -> (1, seq_len, 3, H, W) normalized to [0, 1]
        frames_tensor = torch.from_numpy(face_crops).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)

        # Audio tensor: (1, time_steps, n_mels)
        audio_tensor = torch.from_numpy(audio_spec).unsqueeze(0).to(self.device)

        # Step 6: Model Forward Pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(frames_tensor, audio_tensor if has_audio else None, has_audio=has_audio)

        probs = output["probabilities"][0].cpu().numpy()
        real_prob = float(probs[0]) * 100.0
        fake_prob = float(probs[1]) * 100.0

        predicted_class = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = fake_prob if predicted_class == "FAKE" else real_prob

        # Frame-level predictions & scores
        frame_probs = output["frame_probabilities"][0].cpu().numpy()  # (seq_len, 2)
        frame_results = []
        for i, (f_prob, is_detected) in enumerate(zip(frame_probs, face_res["detection_mask"])):
            f_fake_score = float(f_prob[1]) * 100.0
            frame_results.append({
                "frame_index": extract_res["indices"][i] if i < len(extract_res["indices"]) else i,
                "face_detected": bool(is_detected),
                "fake_score": round(f_fake_score, 2),
                "prediction": "FAKE" if f_fake_score >= 50.0 else "REAL"
            })

        processing_time = round(time.time() - start_time, 2)

        return {
            "success": True,
            "filename": os.path.basename(video_path),
            "video_path": video_path,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "real_probability": round(real_prob, 2),
            "fake_probability": round(fake_prob, 2),
            "frames_analyzed": len(raw_frames),
            "faces_detected": faces_detected,
            "detection_rate": round(face_res["detection_rate"] * 100.0, 1),
            "modality_used": output["modality_used"],
            "has_audio": has_audio,
            "audio_duration_sec": audio_res["duration_sec"],
            "video_metadata": metadata,
            "frame_results": frame_results,
            "face_crops_preview": face_crops[:min(6, len(face_crops))],
            "raw_frames_preview": raw_frames[:min(6, len(raw_frames))],
            "processing_time": processing_time,
            "model_loaded": self.model_loaded
        }
