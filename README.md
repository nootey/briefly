# Briefly

Drop an audio or video file in - or paste a link - and get a raw transcript and a short summary
out. 
Stateless - no database, no crawling, no follow-up logic. Just data in, data out.

Useful for getting transcripts from audio files, such as podcasts - and optional summaries, for easier consumption.

## Setup

This project uses [uv](https://docs.astral.sh/uv/):

```shell
uv sync --extra cuda   # drop the extra if you have no NVIDIA GPU
cp .env.example .env
```

Local transcription needs nothing else installed - audio is decoded through PyAV, which carries its
own ffmpeg. Summarizing against a local endpoint needs that server running - e.g.
[Ollama](https://ollama.com/) with the model pulled via `ollama pull <name>`.

The first local run downloads the model weights to `~/.cache/huggingface` (`large-v3` is ~3 GB).

```shell
uv run --extra cuda python -c "import torch; print(torch.cuda.is_available())"
```

### Configuration

`.env` holds two keys, both optional:

| Variable | Needed when |
| --- | --- |
| `OPENAI_API_KEY` | `transcribe.provider` is `openai`. Unused for local transcription. |
| `SUMMARY_MODEL_API_KEY` | The provider `summarize.base_url` points at requires a key. Leave empty for Ollama and friends. |

Rest of the configuration is handled by `config.yaml`. The CLI takes four arguments:

| Argument | Description |
| --- | --- |
| `-f`, `--file INPUT_FILE` | Audio/video file to process. A bare name is looked up in `paths.input_dir` (`data/input`). |
| `-u`, `--url URL` | Link to download the audio from. Either this or `--file` is required, not both. |
| `--config PATH` | Path to the config file. Defaults to `config.yaml`. |
| `--transcribe-only` | Write the transcript and skip the summary. |

To change models, endpoints or devices, edit `config.yaml` - or keep variants side by side and
point at them per run:

```shell
uv run --extra cuda python -m main --file meeting.mp3 --config config.local.yaml
uv run --extra cuda python -m main --file meeting.mp3 --transcribe-only
```

## Usage

```shell
uv run --extra cuda python -m main --file meeting.mp3   # looked up in data/input
make file=meeting.mp3                                   # any var= is forwarded as a flag
make file=meeting.mp3 transcribe-only=1
```

`--url` takes a link instead, downloads the audio into `data/input`, and carries on from there.
YouTube is the one that gets tested; yt-dlp reaches most other video sites too, and a link that
points straight at a media file is downloaded directly:

```shell
uv run --extra cuda python -m main --url 'https://www.youtube.com/watch?v=jNQXAC9IVRw'
make url='https://www.youtube.com/watch?v=jNQXAC9IVRw'
```

For consistency, quote the url before passing it. A plain `watch?v=...` link needs no quotes. One with an `&` in it does, 
because `&` tells your shell to start a background command, and it cuts the link there
before make or briefly ever see it.

The file keeps its title and video ID (`Me_at_the_zoo-jNQXAC9IVRw.webm`), 
which means a second run of the same link is idempotent.

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
time - set it to `false` if you don't need it.

The `turbo` models are the exception: they aren't trained for translation and return the source
language even when asked, so the English copy is skipped (with a note) rather than written with
the wrong contents. Use `large-v3` when you need translation.

## Local

```shell
make lint
make test
uv run --group dev pre-commit install
```
