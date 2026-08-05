import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import TranscribeConfig

# Formats the OpenAI transcription endpoint accepts directly.
API_FORMATS = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}

# Whisper's models are all trained on 16 kHz mono, and decode_audio resamples to it.
SAMPLE_RATE = 16000

# On a terminal, progress is redrawn in place this often. Anywhere else — a pipe,
PROGRESS_MIN_INTERVAL = 0.5
PROGRESS_STEP_SECONDS = 300
# Wide enough that a redraw covers whatever the previous one left behind.
PROGRESS_WIDTH = 64


@dataclass
class Result:
    text: str
    language: str
    english: str | None = None


def run(audio_path: Path, config: TranscribeConfig, api_key: str | None = None) -> Result:
    if config.provider == "local":
        return _local(audio_path, config)
    if config.provider == "openai":
        return _openai(audio_path, config, api_key)
    raise ValueError(f"unknown transcribe provider: {config.provider!r} (use 'local' or 'openai')")

def load_torch():
    """torch ships as the optional `cuda` extra; without it there is no GPU path."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_device(requested: str) -> str:
    torch = load_torch()
    has_cuda = torch is not None and torch.cuda.is_available()

    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested == "cuda" and not has_cuda:
        missing = "torch is not installed" if torch is None else "torch reports no CUDA GPU"
        raise RuntimeError(
            f"config asks for device: cuda but {missing}. Install the GPU extra with "
            f"`uv sync --extra cuda`, or use device: auto to fall back to the CPU."
        )
    return requested


def resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    # float16 halves both VRAM and time on any modern GPU. CTranslate2 has no
    # float16 kernels for the CPU, where int8 is the fast path instead.
    return "float16" if device == "cuda" else "int8"


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _register_cuda_libraries() -> None:
    torch = load_torch()
    if sys.platform != "win32" or torch is None:
        return

    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))


def _draw_progress(done_seconds: float, audio_seconds: float, elapsed: float, live: bool) -> None:
    fraction = min(done_seconds / audio_seconds, 1.0)
    speed = done_seconds / elapsed if elapsed > 0 else 0
    remaining = (audio_seconds - done_seconds) / speed if speed > 0 else 0
    line = (
        f"    {fraction:4.0%}  {format_duration(done_seconds)} of "
        f"{format_duration(audio_seconds)}  {speed:.1f}x  eta {format_duration(remaining)}"
    )
    print(line.ljust(PROGRESS_WIDTH), end="\r" if live else "\n", flush=True)


def _decode_pass(pipeline, audio, audio_seconds: float, task: str, **options) -> tuple[str, str]:
    segments, info = pipeline.transcribe(audio, task=task, **options)

    live = sys.stdout.isatty()
    started = time.monotonic()
    parts: list[str] = []
    last_drawn, next_mark = 0.0, PROGRESS_STEP_SECONDS

    for segment in segments:
        parts.append(segment.text.strip())
        if not audio_seconds:
            continue
        now = time.monotonic()
        due = now - last_drawn >= PROGRESS_MIN_INTERVAL if live else segment.end >= next_mark
        if not due:
            continue
        last_drawn, next_mark = now, segment.end + PROGRESS_STEP_SECONDS
        _draw_progress(segment.end, audio_seconds, now - started, live)

    if live and audio_seconds:
        # Leave the line reading 100% rather than wherever the last segment ended,
        # then break out of it so the next print starts on a clean row.
        _draw_progress(audio_seconds, audio_seconds, time.monotonic() - started, live)
        print()

    return " ".join(part for part in parts if part), str(info.language)


def _transcribe_local(audio_path: Path, config: TranscribeConfig) -> Result:
    local = config.local
    device = resolve_device(local.device)
    compute_type = resolve_compute_type(device, local.compute_type)

    _register_cuda_libraries()
    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

    started = time.monotonic()
    print(f"  loading faster-whisper '{local.model}' on {device} ({compute_type}) ...", flush=True)
    try:
        model = WhisperModel(local.model, device=device, compute_type=compute_type)
    except OSError as exc:
        raise RuntimeError(
            f"could not load the '{local.model}' weights: {exc}. If this was a "
            f"download, check your connection and retry — a half-finished file is "
            f"discarded rather than reused."
        ) from exc

    # The batched pipeline is where the speed comes from: it splits the audio on
    # silence and decodes several of those windows at once, instead of walking
    # 30-second windows one after another.
    pipeline = BatchedInferencePipeline(model=model)
    print(f"  model ready in {time.monotonic() - started:.1f}s", flush=True)

    # Decode up front so we can report the length before the slow part starts.
    print(f"  decoding {audio_path.name} ...", flush=True)
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    audio_seconds = len(audio) / SAMPLE_RATE
    print(f"  audio is {format_duration(audio_seconds)}, transcribing ...", flush=True)

    # Shared by both decoding passes below.
    options = {
        "initial_prompt": config.prompt or None,
        "beam_size": local.beam_size,
        "batch_size": local.batch_size,
    }

    started = time.monotonic()
    text, language = _decode_pass(
        pipeline,
        audio,
        audio_seconds,
        task="transcribe",
        language=None if config.language == "auto" else config.language,
        **options,
    )
    elapsed = time.monotonic() - started

    print(f"  detected language: {language}")
    speed = f", {audio_seconds / elapsed:.1f}x realtime" if elapsed > 0 else ""
    print(f"  transcribed in {format_duration(elapsed)}{speed}")

    english = None
    if config.translate_to_english and language != "en":
        if "turbo" in local.model:
            # Per whisper's README, turbo is not trained for translation and hands
            # back the source language regardless. Writing that as .en would be a
            # silent lie, so skip it rather than produce a mislabelled file.
            print(
                f"  audio is {language}, but '{local.model}' cannot translate — it "
                f"returns the source language even when asked. Skipping the English "
                f"copy; use large-v3 for translation."
            )
        else:
            # A second pass over the same audio. Whisper's translate task always
            # targets English, so there is nothing to configure.
            print(f"  audio is {language}, translating to English ...", flush=True)
            started = time.monotonic()
            english, _ = _decode_pass(
                pipeline, audio, audio_seconds, task="translate", language=language, **options
            )
            print(f"  translated in {format_duration(time.monotonic() - started)}")

    return Result(text=text, language=language, english=english)


def _worker(audio_path: Path, config: TranscribeConfig, result: "mp.Queue") -> None:
    try:
        result.put((True, _transcribe_local(audio_path, config)))
    except Exception as exc:  # noqa: BLE001 - relayed to the parent verbatim
        result.put((False, f"{type(exc).__name__}: {exc}"))


def _local(audio_path: Path, config: TranscribeConfig) -> Result:
    timeout = config.local.timeout_minutes * 60
    if timeout <= 0:
        return _transcribe_local(audio_path, config)

    ctx = mp.get_context("spawn")
    channel = ctx.Queue()
    process = ctx.Process(target=_worker, args=(audio_path, config, channel), daemon=True)
    process.start()

    deadline = time.monotonic() + timeout
    while True:
        try:
            # Drain the queue before joining: a child blocked writing to a full
            # pipe while the parent waits on join() would deadlock.
            ok, payload = channel.get(timeout=0.5)
            break
        except queue.Empty:
            if not process.is_alive():
                try:  # it may have queued a result in the instant before exiting
                    ok, payload = channel.get(timeout=1)
                    break
                except queue.Empty:
                    raise RuntimeError(
                        f"the transcription process exited without a result "
                        f"(exit code {process.exitcode}) — it likely ran out of memory"
                    ) from None
            if time.monotonic() >= deadline:
                process.terminate()
                process.join(5)
                raise RuntimeError(
                    f"transcription exceeded transcribe.local.timeout_minutes "
                    f"({config.local.timeout_minutes:g} min). Raise it in config.yaml, "
                    f"or set it to 0 to wait indefinitely. Batched large-v3 runs at "
                    f"roughly 20x realtime on a GPU and far slower on a CPU, and "
                    f"translate_to_english adds a second pass on top."
                )

    process.join(10)
    if not ok:
        raise RuntimeError(payload)
    return payload


def _openai(audio_path: Path, config: TranscribeConfig, api_key: str | None) -> Result:
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in .env (required by transcribe.provider: openai)"
        )
    if audio_path.suffix.lower() not in API_FORMATS:
        raise ValueError(
            f"{audio_path.suffix} is not accepted by the OpenAI API; "
            f"convert it or transcribe locally"
        )

    from openai import OpenAI

    print(f"  calling OpenAI '{config.openai.model}'")
    client = OpenAI(api_key=api_key)

    detailed = config.openai.model == "whisper-1"
    with audio_path.open("rb") as fh:
        response = client.audio.transcriptions.create(
            model=config.openai.model,
            file=fh,
            language=None if config.language == "auto" else config.language,
            prompt=config.prompt or None,
            response_format="verbose_json" if detailed else "text",
        )

    if detailed:
        text, language = response.text.strip(), str(response.language)
    else:
        text = str(response).strip()  # response_format="text" yields a bare string
        language = config.language if config.language != "auto" else "unknown"
    print(f"  detected language: {language}")

    english = None
    if config.translate_to_english and language.lower() not in {"en", "english", "unknown"}:
        if not detailed:
            print(f"  note: the API translates with whisper-1 only, so '{config.openai.model}' "
                  f"gets no English copy")
        else:
            print(f"  audio is {language}, translating to English ...")
            with audio_path.open("rb") as fh:
                english = str(
                    client.audio.translations.create(
                        model="whisper-1", file=fh, response_format="text"
                    )
                ).strip()

    return Result(text=text, language=language, english=english)
