import pytest

from src import config as cfg


def write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_file_is_absent(tmp_path):
    config = cfg.load(tmp_path / "nowhere.yaml")

    assert config.transcribe.provider == "local"
    assert config.transcribe.local.model == "large-v3"
    assert config.summarize.chunk_chars == 30000
    assert config.paths.input_dir == "data/input"


def test_nested_values_are_read(tmp_path):
    path = write(
        tmp_path,
        """
        transcribe:
          provider: openai
          local:
            model: turbo
            beam_size: 1
        paths:
          output_dir: out
        """,
    )

    config = cfg.load(path)

    assert config.transcribe.provider == "openai"
    assert config.transcribe.local.model == "turbo"
    assert config.transcribe.local.beam_size == 1
    assert config.paths.output_dir == "out"
    # Untouched siblings keep their defaults rather than being dropped.
    assert config.transcribe.local.batch_size == 16
    assert config.paths.input_dir == "data/input"


def test_unknown_key_is_rejected(tmp_path):
    path = write(tmp_path, "transcribe:\n  modle: turbo\n")

    with pytest.raises(ValueError, match="unknown key"):
        cfg.load(path)


def test_unknown_nested_key_names_its_section(tmp_path):
    path = write(tmp_path, "transcribe:\n  local:\n    gpu: yes\n")

    with pytest.raises(ValueError, match=r"transcribe\.local"):
        cfg.load(path)


def test_non_mapping_section_is_rejected(tmp_path):
    path = write(tmp_path, "transcribe: large-v3\n")

    with pytest.raises(TypeError, match="must be a mapping"):
        cfg.load(path)


def test_non_mapping_top_level_is_rejected(tmp_path):
    path = write(tmp_path, "- one\n- two\n")

    with pytest.raises(ValueError, match="mapping at the top level"):
        cfg.load(path)


def test_empty_value_means_default(tmp_path):
    path = write(tmp_path, "transcribe:\n  provider:\n  local:\n")

    config = cfg.load(path)

    assert config.transcribe.provider == "local"
    assert config.transcribe.local.model == "large-v3"


def test_int_is_coerced_for_float_fields(tmp_path):
    path = write(tmp_path, "transcribe:\n  local:\n    timeout_minutes: 30\n")

    timeout = cfg.load(path).transcribe.local.timeout_minutes

    assert isinstance(timeout, float)
    assert timeout == 30.0


def test_overrides_win_over_the_file(tmp_path):
    path = write(tmp_path, "transcribe:\n  provider: local\n")

    config = cfg.load(path, {"transcribe.provider": "openai", "transcribe.local.model": "small"})

    assert config.transcribe.provider == "openai"
    assert config.transcribe.local.model == "small"


def test_none_overrides_are_ignored(tmp_path):
    path = write(tmp_path, "transcribe:\n  provider: openai\n")

    config = cfg.load(path, {"transcribe.provider": None})

    assert config.transcribe.provider == "openai"


def test_transcribe_model_follows_the_provider(tmp_path):
    path = write(
        tmp_path,
        """
        transcribe:
          provider: openai
          local:
            model: turbo
          openai:
            model: whisper-1
        """,
    )

    config = cfg.load(path)
    assert config.transcribe_model == "whisper-1"

    config.transcribe.provider = "local"
    assert config.transcribe_model == "turbo"


def test_api_keys_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SUMMARY_MODEL_API_KEY", "")

    config = cfg.load(tmp_path / "nowhere.yaml")

    assert config.openai_api_key == "sk-test"
    # An empty variable is treated as unset, not as an empty key.
    assert config.summary_api_key is None


def test_fetch_defaults_and_overrides(tmp_path):
    assert cfg.load(tmp_path / "nowhere.yaml").fetch.format == "bestaudio/best"

    path = write(
        tmp_path,
        """
        fetch:
          format: worstaudio
          noplaylist: false
          cookies_file: ~/cookies.txt
        """,
    )

    config = cfg.load(path)
    assert config.fetch.format == "worstaudio"
    assert config.fetch.noplaylist is False
    assert config.fetch.cookies_file == "~/cookies.txt"
    # Untouched keys keep their defaults.
    assert config.fetch.timeout_seconds == 60


def test_fetch_rejects_an_unknown_key(tmp_path):
    path = write(tmp_path, "fetch:\n  audio_format: mp3\n")

    with pytest.raises(ValueError, match="unknown key"):
        cfg.load(path)
