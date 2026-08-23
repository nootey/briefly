import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_FILE = "config.yaml"


@dataclass
class LocalWhisper:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    batch_size: int = 16
    beam_size: int = 5
    timeout_minutes: float = 60


@dataclass
class OpenAIWhisper:
    model: str = "whisper-1"


@dataclass
class TranscribeConfig:
    provider: str = "local"
    language: str = "auto"
    prompt: str = ""
    translate_to_english: bool = True
    local: LocalWhisper = field(default_factory=LocalWhisper)
    openai: OpenAIWhisper = field(default_factory=OpenAIWhisper)


@dataclass
class SummarizeConfig:
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    chunk_chars: int = 30000
    prompt: str = ""


@dataclass
class FetchConfig:
    format: str = "bestaudio/best"
    filename_template: str = "%(title).80s-%(id)s.%(ext)s"
    noplaylist: bool = True
    cookies_file: str = ""
    timeout_seconds: float = 60


@dataclass
class PathsConfig:
    input_dir: str = "data/input"
    output_dir: str = "data/output"


@dataclass
class Config:
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    summarize: SummarizeConfig = field(default_factory=SummarizeConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def openai_api_key(self) -> str | None:
        """Only needed when transcribing through the OpenAI API."""
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def summary_api_key(self) -> str | None:
        """Whichever provider summarize.base_url points at."""
        return os.getenv("SUMMARY_MODEL_API_KEY") or None

    @property
    def transcribe_model(self) -> str:
        return getattr(self.transcribe, self.transcribe.provider).model


def _build(cls, data, where: str):
    """Instantiate a (possibly nested) config dataclass from plain YAML data."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{where} must be a mapping in {CONFIG_FILE}")

    known = {f.name: f.type for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown key(s) under {where}: {', '.join(sorted(unknown))}")

    values = {}
    for key, value in data.items():
        expected = known[key]
        if is_dataclass(expected):
            values[key] = _build(expected, value, f"{where}.{key}")
        elif value is None:
            continue  # an empty YAML value means "use the default"
        else:
            values[key] = float(value) if expected is float and isinstance(value, int) else value
    return cls(**values)


def load(path: str | Path = CONFIG_FILE, overrides: dict | None = None) -> Config:
    """Read .env and config.yaml, then apply CLI overrides on top."""
    load_dotenv()

    path = Path(path)
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")

    config = _build(Config, data, path.name)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        *parents, key = dotted.split(".")
        target = config
        for part in parents:
            target = getattr(target, part)
        setattr(target, key, value)

    return config
