import json

import pytest

from speech_to_text import Segment, Transcription
from speech_to_text.formats import render

T = Transcription(
    text="Hello world. Second line.",
    segments=[Segment(0.0, 1.5, "Hello world."), Segment(3661.25, 3663.0, "Second line.")],
    language="en", duration=3663.0, engine="whisper", model="base",
)


def test_txt():
    assert render(T, "txt") == "Hello world. Second line.\n"


def test_srt():
    out = render(T, "srt")
    assert out.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello world.\n")
    assert "2\n01:01:01,250 --> 01:01:03,000\nSecond line." in out


def test_vtt():
    out = render(T, "vtt")
    assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world.")


def test_json_roundtrip():
    data = json.loads(render(T, "json"))
    assert data["language"] == "en"
    assert data["segments"][1] == {"start": 3661.25, "end": 3663.0, "text": "Second line."}


def test_unknown_format():
    with pytest.raises(ValueError):
        render(T, "doc")
