from speech_to_text import Segment, Transcription, cli


class FakeEngine:
    def transcribe(self, path, language=None, translate=False):
        return Transcription("hi there", [Segment(0, 1, "hi there")], "en", 1.0, "fake", "x", 0.1)


def test_cli_writes_files(monkeypatch, tmp_path, wav_file):
    monkeypatch.setattr(cli, "get_engine", lambda *a, **k: FakeEngine())
    assert cli.main([wav_file, "-f", "srt", "-o", str(tmp_path / "out")]) == 0
    assert "hi there" in (tmp_path / "out" / "tone.srt").read_text()


def test_cli_stdout_and_missing_file(monkeypatch, capsys, wav_file):
    monkeypatch.setattr(cli, "get_engine", lambda *a, **k: FakeEngine())
    assert cli.main([wav_file, "missing.wav"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "hi there\n"
    assert "file not found" in captured.err


def test_list_models(capsys):
    assert cli.main(["--list-models"]) == 0
    assert "large-v3" in capsys.readouterr().out
