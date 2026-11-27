"""Speech recognition engines behind one small interface.

Three engines are available:

* ``whisper``  - OpenAI Whisper via faster-whisper (default; multilingual,
  timestamps, translation to English, runs offline on CPU or GPU).
* ``wav2vec2`` - Facebook wav2vec 2.0 CTC models via Hugging Face
  transformers (English, offline).
* ``google``   - Google Web Speech API via the SpeechRecognition library
  (online, no model download).

Heavy dependencies are imported lazily, so you only need to install the
packages for the engines you actually use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from .audio import SAMPLE_RATE, chunk, load_audio

ProgressFn = Optional[Callable[[float], None]]


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    duration: float = 0.0
    engine: str = ""
    model: str = ""
    elapsed: float = 0.0


class Engine:
    name = ""
    default_model = ""
    models: tuple[str, ...] = ()

    def __init__(self, model: Optional[str] = None):
        self.model_name = model or self.default_model

    def transcribe(
        self,
        path: str,
        language: Optional[str] = None,
        translate: bool = False,
        progress: ProgressFn = None,
    ) -> Transcription:
        started = time.perf_counter()
        result = self._transcribe(path, language, translate, progress)
        result.engine, result.model = self.name, self.model_name
        result.elapsed = time.perf_counter() - started
        return result

    def _transcribe(self, path, language, translate, progress) -> Transcription:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Whisper (faster-whisper)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _whisper_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device=device, compute_type=compute_type)


class WhisperEngine(Engine):
    name = "whisper"
    default_model = "base"
    models = (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v3", "large-v3-turbo", "distil-large-v3",
    )

    def __init__(self, model=None, device="auto", compute_type="default", beam_size=5, vad=True):
        super().__init__(model)
        self.device, self.compute_type = device, compute_type
        self.beam_size, self.vad = beam_size, vad

    def _transcribe(self, path, language, translate, progress):
        model = _whisper_model(self.model_name, self.device, self.compute_type)
        seg_iter, info = model.transcribe(
            path,
            language=language or None,
            task="translate" if translate else "transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad,
        )
        segments = []
        for s in seg_iter:  # segments are generated lazily while decoding
            segments.append(Segment(round(s.start, 3), round(s.end, 3), s.text.strip()))
            if progress and info.duration:
                progress(min(s.end / info.duration, 1.0))
        return Transcription(
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            language=info.language,
            duration=round(info.duration, 3),
        )


# --------------------------------------------------------------------------- #
# wav2vec 2.0 (transformers)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _wav2vec_model(name: str):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device), device


class Wav2Vec2Engine(Engine):
    name = "wav2vec2"
    # Note: the bare "facebook/wav2vec2-large-xlsr-53" checkpoint is
    # pre-trained only and has no CTC head, so it cannot transcribe. These
    # checkpoints are fine-tuned for English ASR.
    default_model = "facebook/wav2vec2-base-960h"
    models = (
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h-lv60-self",
    )
    chunk_seconds = 20.0

    def _transcribe(self, path, language, translate, progress):
        import torch

        processor, model, device = _wav2vec_model(self.model_name)
        audio = load_audio(path)
        duration = len(audio) / SAMPLE_RATE
        segments = []
        for start, window in chunk(audio, self.chunk_seconds):
            if len(window) < SAMPLE_RATE // 10:  # skip <100 ms tails
                continue
            inputs = processor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs.input_values.to(device)).logits
            text = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip().lower()
            end = start + len(window) / SAMPLE_RATE
            if text:
                segments.append(Segment(round(start, 3), round(end, 3), text))
            if progress:
                progress(min(end / duration, 1.0))
        return Transcription(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language="en",
            duration=round(duration, 3),
        )


# --------------------------------------------------------------------------- #
# Google Web Speech (SpeechRecognition)
# --------------------------------------------------------------------------- #
class GoogleEngine(Engine):
    name = "google"
    default_model = "web-speech"
    models = ("web-speech",)
    chunk_seconds = 45.0  # the free endpoint rejects long requests

    def _transcribe(self, path, language, translate, progress):
        import numpy as np
        import speech_recognition as sr

        lang = language or "en-US"
        if len(lang) == 2:
            lang = {"en": "en-US"}.get(lang, lang)
        recognizer = sr.Recognizer()
        audio = load_audio(path)
        duration = len(audio) / SAMPLE_RATE
        segments = []
        for start, window in chunk(audio, self.chunk_seconds):
            pcm = (np.clip(window, -1, 1) * 32767).astype("<i2").tobytes()
            try:
                text = recognizer.recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2), language=lang)
            except sr.UnknownValueError:  # silence or unintelligible
                text = ""
            except sr.RequestError as exc:
                raise RuntimeError(f"Google Web Speech request failed: {exc}") from exc
            end = start + len(window) / SAMPLE_RATE
            if text:
                segments.append(Segment(round(start, 3), round(end, 3), text))
            if progress:
                progress(min(end / duration, 1.0))
        return Transcription(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language=lang,
            duration=round(duration, 3),
        )


ENGINES: dict[str, type[Engine]] = {
    e.name: e for e in (WhisperEngine, Wav2Vec2Engine, GoogleEngine)
}


def get_engine(name: str = "whisper", model: Optional[str] = None, **kwargs) -> Engine:
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f"Unknown engine {name!r}; choose from {', '.join(ENGINES)}") from None
    return cls(model, **kwargs)
