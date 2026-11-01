"""
Unit tests for Preprocessing Modules:
- Video Frame Extraction
- Face Detection and Preprocessing
- Audio Spectrogram Extraction
"""

import os
import cv2
import numpy as np
import pytest

from src.preprocessing.extract_frames import validate_video_file, VideoFrameExtractor, frame_extract
from src.preprocessing.detect_faces import FaceDetector
from src.preprocessing.audio_extract import AudioExtractor, DEFAULT_MAX_AUDIO_FRAMES, DEFAULT_N_MELS


@pytest.fixture(scope="session")
def sample_test_video(tmp_path_factory):
    """
    Creates a temporary valid 1-second test video clip with moving facial shapes.
    """
    temp_dir = tmp_path_factory.mktemp("test_videos")
    video_path = os.path.join(temp_dir, "test_clip.mp4")

    width, height = 320, 240
    fps = 15
    num_frames = 20

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        # Draw a synthetic face circle
        center = (160 + int(10 * np.sin(i / 2)), 120)
        cv2.circle(frame, center, 45, (220, 180, 150), -1)  # Face
        cv2.circle(frame, (center[0] - 15, center[1] - 10), 6, (40, 40, 40), -1)  # Left eye
        cv2.circle(frame, (center[0] + 15, center[1] - 10), 6, (40, 40, 40), -1)  # Right eye
        cv2.ellipse(frame, (center[0], center[1] + 15), (15, 8), 0, 0, 180, (200, 50, 50), 3)  # Mouth
        out.write(frame)

    out.release()
    return video_path


def test_validate_video_file(sample_test_video):
    # Non-existent file
    valid, msg = validate_video_file("non_existent.mp4")
    assert not valid
    assert "does not exist" in msg

    # Invalid extension
    valid, msg = validate_video_file("sample.txt")
    assert not valid
    assert "Unsupported format" in msg

    # Valid video
    valid, msg = validate_video_file(sample_test_video)
    assert valid
    assert "Valid" in msg


def test_frame_extractor(sample_test_video, tmp_path):
    extractor = VideoFrameExtractor(target_size=(224, 224))
    meta = extractor.get_video_metadata(sample_test_video)
    assert meta["total_frames"] == 20
    assert meta["width"] == 320
    assert meta["height"] == 240

    save_dir = os.path.join(tmp_path, "frames")
    result = extractor.extract_sequence(sample_test_video, sequence_length=6, save_dir=save_dir)

    assert result["count"] == 6
    assert len(result["frames"]) == 6
    assert result["frames"][0].shape == (224, 224, 3)
    assert os.path.exists(save_dir)
    assert len(result["saved_paths"]) == 6


def test_generator_frame_extract(sample_test_video):
    frames = list(frame_extract(sample_test_video))
    assert len(frames) == 20
    assert frames[0].shape == (240, 320, 3)


def test_face_detector(sample_test_video):
    extractor = VideoFrameExtractor(target_size=(224, 224))
    res = extractor.extract_sequence(sample_test_video, sequence_length=4)
    frames = res["frames"]

    detector = FaceDetector(target_size=(224, 224), padding_ratio=0.2)
    processed = detector.process_sequence(frames)

    assert processed["total_count"] == 4
    assert processed["face_crops"].shape == (4, 224, 224, 3)
    assert len(processed["bboxes"]) == 4
    assert len(processed["detection_mask"]) == 4


def test_audio_extractor_non_existent():
    extractor = AudioExtractor()
    res = extractor.extract_spectrogram("non_existent_audio.mp4")
    assert res["has_audio"] is False
    assert res["spectrogram"].shape == (DEFAULT_MAX_AUDIO_FRAMES, DEFAULT_N_MELS)


def test_audio_extractor_silent_video(sample_test_video):
    extractor = AudioExtractor()
    res = extractor.extract_spectrogram(sample_test_video)
    assert res["spectrogram"].shape == (DEFAULT_MAX_AUDIO_FRAMES, DEFAULT_N_MELS)
    assert isinstance(res["has_audio"], bool)
