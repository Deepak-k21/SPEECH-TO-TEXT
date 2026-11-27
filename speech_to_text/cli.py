"""Command line interface: ``python -m speech_to_text <files> [options]``."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .engines import ENGINES, get_engine
from .formats import FORMATS, render


def record_microphone(seconds: float, sr: int = 16_000) -> str:
    """Record from the default microphone and return a temporary WAV path."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit("Microphone recording needs: pip install sounddevice soundfile")
    print(f"Recording for {seconds:g}s... speak now.", file=sys.stderr)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, audio, sr)
    print("Recording finished, transcribing...", file=sys.stderr)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speech-to-text",
        description="Transcribe audio or video files (or your microphone) to text.",
    )
    p.add_argument("files", nargs="*", help="audio/video files to transcribe")
    p.add_argument("-e", "--engine", choices=list(ENGINES), default="whisper")
    p.add_argument("-m", "--model", help="model name (default depends on engine)")
    p.add_argument("-l", "--language", help="language code, e.g. en, hi, ta (auto-detect if omitted)")
    p.add_argument("--translate", action="store_true", help="translate speech to English (whisper only)")
    p.add_argument("-f", "--format", choices=FORMATS, default="txt", help="output format")
    p.add_argument("-o", "--output-dir", type=Path, help="write <name>.<format> files here instead of stdout")
    p.add_argument("--mic", type=float, metavar="SECONDS", help="record from the microphone for SECONDS")
    p.add_argument("--device", default="auto", help="whisper device: auto, cpu or cuda")
    p.add_argument("--list-models", action="store_true", help="list suggested models per engine and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        for name, cls in ENGINES.items():
            print(f"{name} (default: {cls.default_model})")
            for m in cls.models:
                print(f"  {m}")
        return 0

    files = list(args.files)
    if args.mic:
        files.append(record_microphone(args.mic))
    if not files:
        build_parser().error("give at least one file or use --mic SECONDS")
    if args.translate and args.engine != "whisper":
        build_parser().error("--translate is only supported by the whisper engine")

    kwargs = {"device": args.device} if args.engine == "whisper" else {}
    engine = get_engine(args.engine, args.model, **kwargs)

    status = 0
    for f in files:
        if not Path(f).is_file():
            print(f"error: file not found: {f}", file=sys.stderr)
            status = 1
            continue
        result = engine.transcribe(f, language=args.language, translate=args.translate)
        output = render(result, args.format)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / f"{Path(f).stem}.{args.format}"
            out.write_text(output, encoding="utf-8")
            print(f"{f} -> {out}  ({result.duration:.1f}s audio in {result.elapsed:.1f}s)", file=sys.stderr)
        else:
            sys.stdout.write(output)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
