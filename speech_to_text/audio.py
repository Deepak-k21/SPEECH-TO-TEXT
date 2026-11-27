"""Audio loading helpers shared by the engines."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file as mono float32 at ``sr`` Hz.

    librosa reads WAV/FLAC/OGG natively and falls back to ffmpeg (via
    audioread) for MP3, M4A, WEBM and video containers.
    """
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def chunk(audio: np.ndarray, seconds: float, sr: int = SAMPLE_RATE):
    """Yield ``(start_seconds, samples)`` windows of at most ``seconds`` each."""
    step = max(1, int(seconds * sr))
    for start in range(0, len(audio), step):
        yield start / sr, audio[start : start + step]
