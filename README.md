# Briefly

Drop an audio or video file in, get a raw transcript and a short summary out. Stateless — no
database, no crawling, no follow-up logic. Just data in, data out.

## Setup

This project uses [uv](https://docs.astral.sh/uv/):

```shell
uv sync --extra cuda   # drop the extra if you have no NVIDIA GPU
cp .env.example .env
```

Local transcription needs nothing else installed — audio is decoded through PyAV, which carries its
own ffmpeg. Summarizing against a local endpoint needs that server running — e.g.
[Ollama](https://ollama.com/) with the model pulled via `ollama pull <name>`.

The first local run downloads the model weights to `~/.cache/huggingface` (`large-v3` is ~3 GB).

`.env` holds two keys, both optional:

| Variable | Needed when |
| --- | --- |
| `OPENAI_API_KEY` | `transcribe.provider` is `openai`. Unused for local transcription. |
| `SUMMARY_MODEL_API_KEY` | The provider `summarize.base_url` points at requires a key. Leave empty for Ollama and friends. |

GPU transcription lives in the optional `cuda` extra, which pulls `torch` from PyTorch's CUDA 12.8
index for the cuBLAS/cuDNN libraries CTranslate2 loads at runtime. It's a multi-gigabyte download,
so a plain `uv sync` leaves it out and everything still runs — on the CPU. Check what you have
with:

```shell
uv run --extra cuda python -c "import torch; print(torch.cuda.is_available())"
```

`uv run` syncs the environment to whatever it's told, so keep `--extra cuda` on every run once
you've installed it — otherwise the next run removes it again. `make run` passes it by default.

## Usage

```shell
uv run --extra cuda python -m main --file meeting.mp3   # looked up in data/input
make file=meeting.mp3                                   # any var= is forwarded as a flag
make file=meeting.mp3 transcribe-only=1
```

Every run gets its own timestamped directory under `data/output`, so nothing is ever overwritten
and repeated runs of the same file sit side by side:

```
data/output/meeting_20260805-142233/
├── transcript.txt      the raw transcript
├── transcript.en.txt   only when the audio was not English
├── summary.md          the summary
└── summary.en.md       only when the audio was not English
```

Whisper can translate non-English speech straight into English, and
`transcribe.translate_to_english` (on by default) uses it: when the detected language isn't
English you get the `.en` pair alongside the originals. English audio is unaffected. The
translation is a second decoding pass over the same audio, so it roughly doubles transcription
time — set it to `false` if you don't need it.

The `turbo` models are the exception: they aren't trained for translation and return the source
language even when asked, so the English copy is skipped (with a note) rather than written with
the wrong contents. Use `large-v3` when you need translation.

Each summary is a `## Summary` section of 250–500 words followed by up to five `## Key Concepts`.
Transcripts longer than `summarize.chunk_chars` are split on paragraph and sentence boundaries,
summarized chunk by chunk, then merged into one summary with a single pass for the concepts.

## Configuration

All settings live in `config.yaml`. The CLI itself takes three arguments:

| Argument | Description |
| --- | --- |
| `-f`, `--file INPUT_FILE` | Required. Audio/video file to process. A bare name is looked up in `paths.input_dir` (`data/input`). |
| `--config PATH` | Path to the config file. Defaults to `config.yaml`. |
| `--transcribe-only` | Write the transcript and skip the summary. |

To change models, endpoints or devices, edit `config.yaml` — or keep variants side by side and
point at them per run:

```shell
uv run --extra cuda python -m main --file meeting.mp3 --config config.local.yaml
uv run --extra cuda python -m main --file meeting.mp3 --transcribe-only
```

The keys worth knowing about — every one is optional and falls back to the value shown in
`config.yaml`:

| Key | What it does |
| --- | --- |
| `transcribe.provider` | `local` (faster-whisper) or `openai` (the API). |
| `transcribe.language` | `auto` detects, or force an ISO code like `sl`. |
| `transcribe.prompt` | Spelling hints fed to Whisper — names, jargon, acronyms. |
| `transcribe.translate_to_english` | Adds the `.en` pair for non-English audio. |
| `transcribe.local.model` | `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `turbo`. |
| `transcribe.local.device` | `auto`, `cuda` or `cpu`. `cuda` errors out if no GPU is present; `auto` falls back. |
| `transcribe.local.compute_type` | `auto` picks `float16` on a GPU, `int8` on a CPU. |
| `transcribe.local.batch_size` | Windows decoded at once — the speed knob, trades VRAM for throughput. |
| `transcribe.local.timeout_minutes` | Abort a run that overruns. `0` waits indefinitely. |
| `summarize.base_url` / `.model` | Any OpenAI-compatible endpoint (see below). |
| `summarize.chunk_chars` | Chunk size for long transcripts. Lower it if the provider returns 413. |
| `summarize.prompt` | Replaces the built-in prompt. Must contain `{transcript}`. |
| `paths.input_dir` / `.output_dir` | Where bare filenames are looked up and runs are written. |

Unknown keys are rejected at load time rather than silently ignored.

### Transcribing through the OpenAI API

Set `transcribe.provider: openai` and put a key in `OPENAI_API_KEY` to skip local inference
entirely. Two caveats: the API only accepts `.flac .m4a .mp3 .mp4 .mpeg .mpga .oga .ogg .wav
.webm`, and translation is a `whisper-1`-only feature — any other `transcribe.openai.model` gets
the transcript but no English copy.

### Swapping summary providers

There is no provider list to pick from — the summarizer talks the OpenAI chat-completions protocol,
so any endpoint that speaks it works. Set `summarize.base_url`, set `summarize.model`, and put that
provider's key in `SUMMARY_MODEL_API_KEY`:

| Provider | `base_url` |
| --- | --- |
| Groq | `https://api.groq.com/openai/v1` |
| OpenAI | `https://api.openai.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Ollama (local) | `http://localhost:11434/v1` — key not needed |

## Development

```shell
make lint
make test
uv run --group dev pre-commit install
```

The tests never touch a model, a GPU or a network call — every slow edge is stubbed — so CI runs
them without the `cuda` extra.
