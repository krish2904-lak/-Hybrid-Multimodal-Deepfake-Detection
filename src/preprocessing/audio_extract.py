"""
Audio Extraction and Spectrogram Preprocessing Module for Multimodal Deepfake Detection
Extracts audio tracks and converts to Mel-Spectrogram features compatible with the audio model.
"""

import os
import shutil
import subprocess
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict, Any, Optional


AUDIO_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 40
DEFAULT_MAX_AUDIO_FRAMES = 130
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


class AudioExtractor:
    """
    Extracts audio tracks from video or audio files and converts them to fixed-shape
    normalized Mel-spectrogram sequences for LSTM / CNN audio processing.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        n_mels: int = DEFAULT_N_MELS,
        max_audio_frames: int = DEFAULT_MAX_AUDIO_FRAMES
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_frames = max_audio_frames

    def extract_spectrogram(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts Mel-spectrogram from video or audio file.
        Returns:
            spectrogram: (max_audio_frames, n_mels) float32 array
            has_audio: bool
            duration_sec: float
            error_message: Optional[str]
        """
        zero_spectrogram = np.zeros((self.max_audio_frames, self.n_mels), dtype=np.float32)

        if not os.path.isfile(file_path):
            return {
                "spectrogram": zero_spectrogram,
                "has_audio": False,
                "duration_sec": 0.0,
                "error_message": f"File does not exist: {file_path}"
            }

        ext = os.path.splitext(file_path)[1].lower()
        y = None
        sr = self.sample_rate

        # 1. Direct audio file reading using soundfile
        if ext in AUDIO_EXTENSIONS:
            try:
                data, in_sr = sf.read(file_path, dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Stereo to mono
                if in_sr != self.sample_rate:
                    data = librosa.resample(data, orig_sr=in_sr, target_sr=self.sample_rate)
                y = data
                sr = self.sample_rate
            except Exception as e:
                return {
                    "spectrogram": zero_spectrogram,
                    "has_audio": False,
                    "duration_sec": 0.0,
                    "error_message": f"Failed reading audio file: {str(e)}"
                }

        # 2. Video file: check if ffmpeg is available to extract audio
        else:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                try:
                    temp_wav = os.path.join(os.path.dirname(file_path), "_temp_extracted_audio.wav")
                    cmd = [
                        ffmpeg_path, "-y", "-i", file_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", str(self.sample_rate),
                        "-ac", "1", temp_wav
                    ]
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8)
                    if res.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
                        y, sr = sf.read(temp_wav, dtype='float32')
                        os.remove(temp_wav)
                    else:
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                        return {
                            "spectrogram": zero_spectrogram,
                            "has_audio": False,
                            "duration_sec": 0.0,
                            "error_message": "No audio stream found in video file."
                        }
                except Exception as e:
                    return {
                        "spectrogram": zero_spectrogram,
                        "has_audio": False,
                        "duration_sec": 0.0,
                        "error_message": f"Video audio extraction failed: {str(e)}"
                    }
            else:
                # Video file without local ffmpeg: gracefully treat as video-only
                return {
                    "spectrogram": zero_spectrogram,
                    "has_audio": False,
                    "duration_sec": 0.0,
                    "error_message": "Video has no extracted audio track (video-only mode active)."
                }

        if y is None or len(y) == 0 or np.all(y == 0):
            return {
                "spectrogram": zero_spectrogram,
                "has_audio": False,
                "duration_sec": 0.0,
                "error_message": "Audio stream is empty or silent."
            }

        try:
            duration_sec = float(len(y)) / float(sr)
            # Compute Mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=1024,
                hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).T

            if mel_db.shape[0] < self.max_audio_frames:
                pad = np.zeros((self.max_audio_frames - mel_db.shape[0], self.n_mels), dtype=np.float32)
                mel_db = np.vstack([mel_db, pad])
            else:
                mel_db = mel_db[:self.max_audio_frames]

            min_val = mel_db.min()
            max_val = mel_db.max()
            if max_val - min_val > 1e-6:
                norm_mel = (mel_db - min_val) / (max_val - min_val)
            else:
                norm_mel = np.zeros_like(mel_db)

            return {
                "spectrogram": norm_mel.astype(np.float32),
                "has_audio": True,
                "duration_sec": duration_sec,
                "error_message": None
            }
        except Exception as e:
            return {
                "spectrogram": zero_spectrogram,
                "has_audio": False,
                "duration_sec": 0.0,
                "error_message": f"Mel-spectrogram extraction error: {str(e)}"
            }
