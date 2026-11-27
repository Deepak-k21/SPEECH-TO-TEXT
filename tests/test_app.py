import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from speech_to_text import Segment, Transcription  # noqa: E402


class FakeEngine:
    def transcribe(self, path, language=None, translate=False, progress=None):
        progress and progress(1.0)
        return Transcription("hello", [Segment(0, 1, "hello")], "en", 1.0, "whisper", "base", 0.5)


def test_ui_builds():
    assert isinstance(app.build_ui(), gr.Blocks)


def test_transcribe_returns_text_segments_and_files(monkeypatch, wav_file):
    monkeypatch.setattr(app, "get_engine", lambda *a, **k: FakeEngine())
    text, rows, files, info = app.transcribe(
        wav_file, "whisper", "base", "English", False, progress=lambda *a, **k: None
    )
    assert text == "hello"
    assert rows == [["0.00", "1.00", "hello"]]
    assert sorted(f.rsplit(".", 1)[1] for f in files) == ["json", "srt", "txt", "vtt"]
    assert "2.0x real time" in info


def test_transcribe_without_audio():
    with pytest.raises(gr.Error):
        app.transcribe(None, "whisper", "base", "English", False, progress=lambda *a, **k: None)
