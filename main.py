import argparse
import sys
from datetime import datetime
from pathlib import Path

from src import config as cfg
from src import summarize, transcribe

# Rough VRAM needs for faster-whisper at float16 and the default batch_size, keyed by the family a model name starts with.
MODEL_VRAM_GIB = {"tiny": 1, "base": 1, "small": 2, "medium": 3, "turbo": 3, "large": 5}


def model_vram_gib(model: str) -> int | None:
    for family, gib in MODEL_VRAM_GIB.items():
        if model.startswith(family):
            return gib
    return None


def check_system_requirements(config: cfg.Config) -> None:
    print("Checking system requirements ...")

    if config.transcribe.provider != "local":
        print("  transcribing via the OpenAI API, nothing needed locally")
        return

    # faster-whisper decodes audio through PyAV, which bundles its own ffmpeg
    # libraries, so there is nothing to check for on PATH.
    device = transcribe.resolve_device(config.transcribe.local.device)
    if device == "cuda":
        # resolve_device only returns cuda once torch has confirmed a GPU, so it is here.
        torch = transcribe.load_torch()

        print(f"  CUDA: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        free, total = torch.cuda.mem_get_info()
        gib = 1024**3
        print(f"  VRAM: {free / gib:.1f} GiB free of {total / gib:.1f} GiB")

        needed = model_vram_gib(config.transcribe.local.model)
        if needed and free / gib < needed:
            print(
                f"  warning: '{config.transcribe.local.model}' wants about {needed:g} GiB "
                f"and only {free / gib:.1f} GiB is free. Close other GPU apps, lower "
                f"transcribe.local.batch_size, or pick a smaller model if it fails to "
                f"allocate."
            )
    else:
        missing = " (torch is not installed — `uv sync --extra cuda` adds the GPU build)"
        print(
            f"  CUDA: not available, falling back to the CPU"
            f"{missing if transcribe.load_torch() is None else ''}. Transcription will "
            f"be much slower — consider a smaller transcribe.local.model such as "
            f"'turbo' or 'small', or set transcribe.provider to 'openai'."
        )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="briefly",
        description="Transcribe an audio/video file and summarize it.",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        metavar="INPUT_FILE",
        help="audio/video file placed in ./data/input. A bare name is looked up.",
    )
    parser.add_argument("--config", default=cfg.CONFIG_FILE, help="path to config.yaml")
    parser.add_argument("--transcribe-only", action="store_true", help="skip the summary")

    return parser.parse_args(argv)


def resolve_input(source: str, input_dir: Path) -> Path:
    """Turn the --file value into a local file, rejecting URLs for now."""
    if source.startswith(("http://", "https://")):
        raise NotImplementedError(
            f"downloading is not implemented yet: {source}. "
            f"Download the file yourself and put it in {input_dir}."
        )

    path = Path(source)
    if not path.exists():
        path = input_dir / source
    if not path.exists():
        raise FileNotFoundError(f"no such file: {source} (looked in {input_dir} too)")
    return path


def process(path: Path, config: cfg.Config, output_dir: Path, args: argparse.Namespace) -> None:
    run_dir = output_dir / f"{path.stem}_{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"Writing this run to {run_dir}")

    print("Transcribing audio ...")
    result = transcribe.run(path, config.transcribe, config.openai_api_key)

    # The original transcript, plus the English one when the audio was not English.
    transcripts = {"transcript.txt": result.text}
    if result.english:
        transcripts["transcript.en.txt"] = result.english

    for name, text in transcripts.items():
        (run_dir / name).write_text(text, encoding="utf-8")
        print(f"File transcribed to {run_dir / name} ({len(text):,} characters)")

    if args.transcribe_only:
        return

    for name, text in transcripts.items():
        summary_name = name.replace("transcript", "summary").replace(".txt", ".md")
        print(f"Summarizing {name} ...")
        summary = summarize.run(text, config.summarize, config.summary_api_key)
        (run_dir / summary_name).write_text(summary, encoding="utf-8")
        print(f"Summary saved to {run_dir / summary_name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Welcome to Briefly ...")

    try:
        config = cfg.load(args.config)
        print(f"Loaded config: {config}")

        # After the config load: what we need depends on the chosen provider.
        check_system_requirements(config)

        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = resolve_input(args.file, Path(config.paths.input_dir))
        print(f"Input file parsed successfully: {path} ({path.stat().st_size / 1024**2:.1f} MiB)")

    except (OSError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


    print(f"Using transcribe model: {config.transcribe.provider}/{config.transcribe_model}")
    if not args.transcribe_only:
        print(f"Using summarize model:  {config.summarize.model} via {config.summarize.base_url}")

    try:
        process(path, config, output_dir, args)
    except Exception as exc:  # noqa: BLE001 - last stop before the traceback reaches the user
        print(f"Processing the file failed: {exc}", file=sys.stderr)
        return 1

    print("\n Transcription complete ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
