"""
Video Frame Extraction Module for Deepfake Detection
Preserves and extends the original frame extraction implementation from legacy codebase.
"""

import os
import cv2
import numpy as np
from typing import Generator, List, Dict, Any, Tuple, Optional


ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}
MAX_DEFAULT_FILE_SIZE_MB = 150


def validate_video_file(file_path: str, max_size_mb: float = MAX_DEFAULT_FILE_SIZE_MB) -> Tuple[bool, str]:
    """
    Validates video file format, existence, and size.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"

    if not os.path.isfile(file_path):
        return False, f"File does not exist: {file_path}"

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed ({max_size_mb} MB)"

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return False, f"Cannot open video file with OpenCV codecs: {file_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        return False, "Video contains 0 readable frames."

    return True, "Valid video file."


def frame_extract(path: str) -> Generator[np.ndarray, None, None]:
    """
    Original frame extraction generator preserved from legacy codebase.
    Yields BGR frames sequentially.
    """
    vid_obj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vid_obj.read()
        if success and image is not None:
            yield image
    vid_obj.release()


class VideoFrameExtractor:
    """
    Extracts frames from video files with configurable sequence length,
    sampling stride, resizing, and directory caching.
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = (224, 224)):
        self.target_size = target_size

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extracts metadata: duration, fps, frame count, dimensions.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "total_frames": 0,
                "fps": 0.0,
                "duration_sec": 0.0,
                "width": 0,
                "height": 0
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "width": width,
            "height": height
        }

    def extract_sequence(
        self,
        video_path: str,
        sequence_length: int = 20,
        stride: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts a sequence of frames evenly spaced or by stride.
        Returns RGB frames, indices, and metadata.
        """
        meta = self.get_video_metadata(video_path)
        total_frames = meta["total_frames"]

        if total_frames <= 0:
            raise ValueError(f"Unable to read frames from video: {video_path}")

        # Determine indices to sample
        if stride is not None and stride > 0:
            indices = list(range(0, total_frames, stride))[:sequence_length]
        else:
            if total_frames <= sequence_length:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int).tolist()

        cap = cv2.VideoCapture(video_path)
        extracted_frames = []
        actual_indices = []

        current_idx = 0
        target_idx_set = set(indices)
        max_target = max(indices) if indices else 0

        while cap.isOpened() and current_idx <= max_target:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in target_idx_set:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.target_size is not None:
                    rgb_frame = cv2.resize(rgb_frame, self.target_size)
                extracted_frames.append(rgb_frame)
                actual_indices.append(current_idx)
            current_idx += 1

        cap.release()

        # If video was shorter than expected, pad or replicate last frame
        if len(extracted_frames) == 0:
            raise RuntimeError(f"Failed to extract any frames from {video_path}")

        # Optionally save to disk
        saved_paths = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            for i, (f, idx) in enumerate(zip(extracted_frames, actual_indices)):
                out_path = os.path.join(save_dir, f"{base_name}_frame_{idx:04d}.jpg")
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, bgr)
                saved_paths.append(out_path)

        return {
            "frames": extracted_frames,
            "indices": actual_indices,
            "saved_paths": saved_paths,
            "metadata": meta,
            "count": len(extracted_frames)
        }
