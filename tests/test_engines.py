import sys
import types
from types import SimpleNamespace

import pytest

from speech_to_text import engines
from speech_to_text.engines import get_engine


def test_unknown_engine():
    with pytest.raises(ValueError):
        get_engine("nope")


def test_default_models():
    assert get_engine("whisper").model_name == "base"
    assert get_engine("wav2vec2").model_name == "facebook/wav2vec2-base-960h"
    assert get_engine("whisper", "small").model_name == "small"


def test_whisper_engine_with_fake_model(monkeypatch, wav_file):
    calls = {}

    class FakeModel:
        def transcribe(self, path, **kw):
            calls.update(kw)
            segs = [SimpleNamespace(start=0.0, end=1.2, text=" Hello "),
                    SimpleNamespace(start=1.2, end=3.0, text=" world ")]
            return iter(segs), SimpleNamespace(language="en", duration=3.0)

    monkeypatch.setattr(engines, "_whisper_model", lambda *a: FakeModel())
    progress = []
    result = get_engine("whisper").transcribe(wav_file, translate=True, progress=progress.append)
    assert result.text == "Hello world"
    assert result.language == "en" and result.duration == 3.0
    assert calls["task"] == "translate" and calls["vad_filter"] is True
    assert progress[-1] == 1.0
    assert result.engine == "whisper" and result.elapsed >= 0


def test_google_engine_with_fake_recognizer(monkeypatch, wav_file):
    class UnknownValueError(Exception): ...
    class RequestError(Exception): ...

    seen = []

    class Recognizer:
        def recognize_google(self, audio, language):
            seen.append((audio.sample_rate, language))
            return "testing one two"

    fake_sr = types.ModuleType("speech_recognition")
    fake_sr.Recognizer = Recognizer
    fake_sr.AudioData = lambda data, rate, width: SimpleNamespace(sample_rate=rate, data=data)
    fake_sr.UnknownValueError, fake_sr.RequestError = UnknownValueError, RequestError
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)

    result = get_engine("google").transcribe(wav_file, language="en")
    assert result.text == "testing one two"
    assert seen == [(16_000, "en-US")]
    assert result.segments[0].end == pytest.approx(3.0, abs=0.01)
