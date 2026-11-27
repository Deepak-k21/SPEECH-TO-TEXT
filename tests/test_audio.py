import numpy as np

from speech_to_text.audio import SAMPLE_RATE, chunk, load_audio


def test_load_resamples_to_mono_16k(wav_file):
    audio = load_audio(wav_file)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert abs(len(audio) - 3 * SAMPLE_RATE) < 10


def test_chunk_covers_all_samples():
    audio = np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32)
    parts = list(chunk(audio, 1.0))
    assert [s for s, _ in parts] == [0.0, 1.0, 2.0]
    assert sum(len(p) for _, p in parts) == len(audio)
