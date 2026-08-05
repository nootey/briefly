import sys
from types import ModuleType, SimpleNamespace

import pytest

from src import transcribe
from src.config import TranscribeConfig


def fake_torch(monkeypatch, *, cuda: bool):
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_run_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown transcribe provider"):
        transcribe.run("audio.mp3", TranscribeConfig(provider="whisper.cpp"))


@pytest.mark.parametrize("cuda, expected", [(True, "cuda"), (False, "cpu")])
def test_auto_device_follows_the_gpu(monkeypatch, cuda, expected):
    fake_torch(monkeypatch, cuda=cuda)

    assert transcribe.resolve_device("auto") == expected


def test_explicit_cuda_without_a_gpu_is_an_error(monkeypatch):
    fake_torch(monkeypatch, cuda=False)

    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        transcribe.resolve_device("cuda")


def no_torch(monkeypatch):
    """Simulate an install without the optional `cuda` extra."""
    monkeypatch.setattr(transcribe, "load_torch", lambda: None)


def test_auto_device_falls_back_to_cpu_without_torch(monkeypatch):
    no_torch(monkeypatch)

    assert transcribe.resolve_device("auto") == "cpu"


def test_explicit_cuda_without_torch_points_at_the_extra(monkeypatch):
    no_torch(monkeypatch)

    with pytest.raises(RuntimeError, match="--extra cuda"):
        transcribe.resolve_device("cuda")


def test_cpu_only_install_needs_no_dll_registration(monkeypatch):
    no_torch(monkeypatch)
    monkeypatch.setattr(transcribe.sys, "platform", "win32")

    transcribe._register_cuda_libraries()  # must not raise


def test_explicit_cpu_is_taken_as_given(monkeypatch):
    fake_torch(monkeypatch, cuda=True)

    assert transcribe.resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(
    "device, requested, expected",
    [
        ("cuda", "auto", "float16"),
        ("cpu", "auto", "int8"),
        ("cpu", "float32", "float32"),
        ("cuda", "int8_float16", "int8_float16"),
    ],
)
def test_compute_type_defaults_per_device(device, requested, expected):
    assert transcribe.resolve_compute_type(device, requested) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0m00s"),
        (9, "0m09s"),
        (61, "1m01s"),
        (3600, "1h00m00s"),
        (3661, "1h01m01s"),
        (7325.9, "2h02m05s"),
    ],
)
def test_format_duration(seconds, expected):
    assert transcribe.format_duration(seconds) == expected


def openai_config(**kwargs):
    return TranscribeConfig(provider="openai", **kwargs)


def test_openai_provider_needs_a_key(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.touch()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        transcribe.run(audio, openai_config(), api_key=None)


def test_openai_provider_rejects_unsupported_formats(tmp_path):
    audio = tmp_path / "clip.mkv"
    audio.touch()

    with pytest.raises(ValueError, match="not accepted by the OpenAI API"):
        transcribe.run(audio, openai_config(), api_key="sk-test")


def fake_openai_client(monkeypatch, *, language: str, calls: list):
    """Install a stand-in `openai` module and record what the code asks it for."""

    def create_transcription(**kwargs):
        calls.append(("transcribe", kwargs))
        return SimpleNamespace(text="  hola mundo  ", language=language)

    def create_translation(**kwargs):
        calls.append(("translate", kwargs))
        return "  hello world  "

    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=create_transcription),
            translations=SimpleNamespace(create=create_translation),
        )
    )
    monkeypatch.setitem(sys.modules, "openai", module)


def test_openai_non_english_audio_gets_a_translation(tmp_path, monkeypatch):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    calls = []
    fake_openai_client(monkeypatch, language="spanish", calls=calls)

    result = transcribe.run(audio, openai_config(), api_key="sk-test")

    assert result.text == "hola mundo"
    assert result.language == "spanish"
    assert result.english == "hello world"
    assert [name for name, _ in calls] == ["transcribe", "translate"]


def test_openai_english_audio_skips_the_second_pass(tmp_path, monkeypatch):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    calls = []
    fake_openai_client(monkeypatch, language="english", calls=calls)

    result = transcribe.run(audio, openai_config(), api_key="sk-test")

    assert result.english is None
    assert [name for name, _ in calls] == ["transcribe"]


def test_openai_translation_can_be_turned_off(tmp_path, monkeypatch):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    calls = []
    fake_openai_client(monkeypatch, language="spanish", calls=calls)

    config = openai_config(translate_to_english=False)
    result = transcribe.run(audio, config, api_key="sk-test")

    assert result.english is None
    assert [name for name, _ in calls] == ["transcribe"]


def test_openai_auto_language_is_sent_as_none(tmp_path, monkeypatch):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    calls = []
    fake_openai_client(monkeypatch, language="english", calls=calls)

    transcribe.run(audio, openai_config(language="auto", prompt=""), api_key="sk-test")

    _, kwargs = calls[0]
    assert kwargs["language"] is None
    assert kwargs["prompt"] is None
    assert kwargs["response_format"] == "verbose_json"


def test_local_worker_relays_failures_to_the_parent(tmp_path, monkeypatch):
    """The timeout path runs the work in a child process; errors must survive the trip."""

    def boom(audio_path, config):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(transcribe, "_transcribe_local", boom)

    class Channel:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    channel = Channel()
    transcribe._worker(tmp_path / "clip.mp3", TranscribeConfig(), channel)

    ok, payload = channel.items[0]
    assert ok is False
    assert payload == "RuntimeError: out of memory"


def test_local_without_a_timeout_runs_in_process(tmp_path, monkeypatch):
    expected = transcribe.Result(text="hi", language="en")
    monkeypatch.setattr(transcribe, "_transcribe_local", lambda path, config: expected)

    config = TranscribeConfig()
    config.local.timeout_minutes = 0

    assert transcribe.run(tmp_path / "clip.mp3", config) is expected
