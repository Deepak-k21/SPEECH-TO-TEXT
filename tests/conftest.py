import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def wav_file(tmp_path):
    """Three seconds of a 440 Hz tone at 44.1 kHz stereo."""
    sr = 44_100
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, np.stack([tone, tone], axis=1), sr)
    return str(path)
