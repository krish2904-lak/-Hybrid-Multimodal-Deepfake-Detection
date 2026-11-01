"""
Face Detection and Preprocessing Module for Deepfake Detection
Uses OpenCV's Haar cascades with adaptive bounding-box padding and
robust center-crop fallback for non-facial or occluded frames.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class FaceDetector:
    """
    Detects and crops faces from video frames with configurable padding,
    target size, and fallback mechanisms.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.25,
        min_face_size: Tuple[int, int] = (40, 40)
    ):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.min_face_size = min_face_size

        # Load OpenCV default Haar cascade for frontal faces
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML not found at {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_in_frame(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects primary face in a single RGB frame.
        Returns cropped face, bounding box (x, y, w, h), and detected flag.
        """
        h, w = rgb_frame.shape[:2]
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Fallback: center crop if no face detected
            crop_size = min(h, w)
            start_y = max(0, (h - crop_size) // 2)
            start_x = max(0, (w - crop_size) // 2)
            crop = rgb_frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
            resized_crop = cv2.resize(crop, self.target_size)
            return {
                "face_crop": resized_crop,
                "bbox": (start_x, start_y, crop_size, crop_size),
                "detected": False,
                "confidence": 0.0
            }

        # Select largest face in frame by area
        best_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, fw, fh = best_face

        # Add adaptive margin / padding
        pad_x = int(fw * self.padding_ratio)
        pad_y = int(fh * self.padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        face_crop = rgb_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = cv2.resize(rgb_frame, self.target_size)
        else:
            face_crop = cv2.resize(face_crop, self.target_size)

        return {
            "face_crop": face_crop,
            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "detected": True,
            "confidence": 0.95
        }

    def process_sequence(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Processes a sequence of RGB frames, returning cropped face images,
        bounding boxes, and detection statistics.
        """
        face_crops = []
        bboxes = []
        detection_mask = []

        for frame in frames:
            result = self.detect_in_frame(frame)
            face_crops.append(result["face_crop"])
            bboxes.append(result["bbox"])
            detection_mask.append(result["detected"])

        detected_count = sum(detection_mask)
        total_count = len(frames)
        detection_rate = detected_count / total_count if total_count > 0 else 0.0

        return {
            "face_crops": np.array(face_crops, dtype=np.uint8),
            "bboxes": bboxes,
            "detection_mask": detection_mask,
            "detected_count": detected_count,
            "total_count": total_count,
            "detection_rate": detection_rate
        }
