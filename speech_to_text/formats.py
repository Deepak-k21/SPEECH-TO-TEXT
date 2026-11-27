"""Export a transcription as plain text, SRT, WebVTT or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict

from .engines import Transcription

FORMATS = ("txt", "srt", "vtt", "json")


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(t: Transcription) -> str:
    return t.text.strip() + "\n"


def to_srt(t: Transcription) -> str:
    blocks = []
    for i, seg in enumerate(t.segments, 1):
        blocks.append(
            f"{i}\n{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_vtt(t: Transcription) -> str:
    lines = ["WEBVTT", ""]
    for seg in t.segments:
        lines.append(f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_json(t: Transcription) -> str:
    return json.dumps(asdict(t), indent=2, ensure_ascii=False) + "\n"


_WRITERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def render(t: Transcription, fmt: str) -> str:
    try:
        return _WRITERS[fmt](t)
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}; choose from {', '.join(FORMATS)}") from None
