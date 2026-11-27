"""Gradio web app: upload a file or record from the microphone, get text.

Run with ``python app.py`` and open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr

from speech_to_text import ENGINES, get_engine
from speech_to_text.formats import render

LANGUAGES = {
    "Auto-detect": None, "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn", "Gujarati": "gu",
    "Urdu": "ur", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Russian": "ru", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
    "Arabic": "ar",
}


def on_engine_change(engine: str):
    cls = ENGINES[engine]
    is_whisper = engine == "whisper"
    return (
        gr.update(choices=list(cls.models), value=cls.default_model),
        gr.update(interactive=is_whisper, value=False),
        gr.update(interactive=engine != "wav2vec2"),
    )


def transcribe(audio_path, engine, model, language, translate, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Upload a file or record something with the microphone first.")
    lang = LANGUAGES.get(language)
    if engine == "google" and lang:
        lang = {"en": "en-US"}.get(lang, lang)

    progress(0, desc="Loading model (first run downloads it)...")
    stt = get_engine(engine, model)
    try:
        result = stt.transcribe(
            audio_path, language=lang, translate=bool(translate) and engine == "whisper",
            progress=lambda p: progress(p, desc="Transcribing..."),
        )
    except Exception as exc:  # surface engine errors in the UI
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    rows = [[f"{s.start:.2f}", f"{s.end:.2f}", s.text] for s in result.segments]
    out_dir = Path(tempfile.mkdtemp(prefix="stt_"))
    stem = Path(audio_path).stem
    files = []
    for fmt in ("txt", "srt", "vtt", "json"):
        p = out_dir / f"{stem}.{fmt}"
        p.write_text(render(result, fmt), encoding="utf-8")
        files.append(str(p))
    speed = result.duration / result.elapsed if result.elapsed else 0
    info = (
        f"**Language:** {result.language or 'n/a'} · **Audio:** {result.duration:.1f}s · "
        f"**Took:** {result.elapsed:.1f}s ({speed:.1f}x real time) · "
        f"**Engine:** {result.engine} / {result.model}"
    )
    return result.text or "(no speech detected)", rows, files, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Speech to Text") as demo:
        gr.Markdown(
            "# 🎙️ Speech to Text\n"
            "Upload an audio/video file or record with your microphone. "
            "Whisper runs fully offline and supports 99 languages."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", label="Audio input"
                )
                engine = gr.Dropdown(list(ENGINES), value="whisper", label="Engine")
                model = gr.Dropdown(
                    list(ENGINES["whisper"].models), value=ENGINES["whisper"].default_model,
                    label="Model", allow_custom_value=True,
                )
                language = gr.Dropdown(list(LANGUAGES), value="Auto-detect", label="Language")
                translate = gr.Checkbox(label="Translate to English (Whisper only)")
                button = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                text = gr.Textbox(label="Transcript", lines=10, buttons=["copy"])
                info = gr.Markdown()
                segments = gr.Dataframe(
                    headers=["Start (s)", "End (s)", "Text"], label="Segments", wrap=True
                )
                downloads = gr.File(label="Download (TXT, SRT, VTT, JSON)", file_count="multiple")

        engine.change(on_engine_change, engine, [model, translate, language])
        button.click(
            transcribe, [audio, engine, model, language, translate],
            [text, segments, downloads, info],
        )
        gr.Markdown(
            "Engines: **whisper** (faster-whisper, offline, multilingual) · "
            "**wav2vec2** (Hugging Face, offline, English) · "
            "**google** (Google Web Speech API, needs internet)."
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Speech to Text web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()
    build_ui().queue().launch(
        server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
