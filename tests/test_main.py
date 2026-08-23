from pathlib import Path

import pytest

import main
from src import fetch, summarize, transcribe
from src.config import Config, FetchConfig


def test_parse_args_requires_a_file_or_a_url():
    with pytest.raises(SystemExit):
        main.parse_args([])


def test_parse_args_refuses_both_a_file_and_a_url():
    with pytest.raises(SystemExit):
        main.parse_args(["-f", "clip.mp3", "-u", "https://youtu.be/abc"])


def test_parse_args_defaults():
    args = main.parse_args(["-f", "clip.mp3"])

    assert args.file == "clip.mp3"
    assert args.url is None
    assert args.config == "config.yaml"
    assert args.transcribe_only is False


def test_parse_args_takes_a_url():
    args = main.parse_args(["--url", "https://youtu.be/abc"])

    assert args.url == "https://youtu.be/abc"
    assert args.file is None


def test_parse_args_flags():
    args = main.parse_args(["--file", "clip.mp3", "--config", "other.yaml", "--transcribe-only"])

    assert args.config == "other.yaml"
    assert args.transcribe_only is True


@pytest.mark.parametrize(
    "model, expected",
    [
        ("large-v3", 5),
        ("turbo", 3),
        ("tiny.en", 1),
        ("distil-large-v3", None),
    ],
)
def test_model_vram_gib(model, expected):
    assert main.model_vram_gib(model) == expected


def test_resolve_input_finds_a_bare_name_in_the_input_dir(tmp_path):
    (tmp_path / "clip.mp3").touch()

    assert main.resolve_input("clip.mp3", tmp_path) == tmp_path / "clip.mp3"


def test_resolve_input_accepts_a_path_as_given(tmp_path):
    nested = tmp_path / "elsewhere"
    nested.mkdir()
    audio = nested / "clip.mp3"
    audio.touch()

    assert main.resolve_input(str(audio), tmp_path) == audio


def test_resolve_input_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="looked in"):
        main.resolve_input("clip.mp3", tmp_path)


def test_resolve_input_downloads_a_url(tmp_path, monkeypatch):
    seen = {}

    def fake_fetch(url, dest_dir, config):
        seen.update(url=url, dest_dir=dest_dir, config=config)
        return dest_dir / "downloaded.webm"

    monkeypatch.setattr(fetch, "run", fake_fetch)
    config = FetchConfig(format="worstaudio")

    path = main.resolve_input("https://youtu.be/abc", tmp_path, config)

    assert path == tmp_path / "downloaded.webm"
    assert seen == {"url": "https://youtu.be/abc", "dest_dir": tmp_path, "config": config}


def test_resolve_input_surfaces_a_failed_download(tmp_path, monkeypatch):
    def boom(url, dest_dir, config):
        raise fetch.FetchError("video unavailable")

    monkeypatch.setattr(fetch, "run", boom)

    with pytest.raises(fetch.FetchError, match="video unavailable"):
        main.resolve_input("https://youtu.be/gone", tmp_path)


@pytest.fixture
def pipeline(monkeypatch):
    """Replace both slow stages with stubs and record what they were handed."""
    seen = {"summarized": []}

    def fake_transcribe(path, config, api_key=None):
        return transcribe.Result(text="hola mundo", language="es", english="hello world")

    def fake_summarize(text, config, api_key=None):
        seen["summarized"].append(text)
        return f"## Summary\n{text}"

    monkeypatch.setattr(transcribe, "run", fake_transcribe)
    monkeypatch.setattr(summarize, "run", fake_summarize)
    return seen


def run_process(tmp_path, argv):
    audio = tmp_path / "clip.mp3"
    audio.touch()
    out = tmp_path / "out"
    args = main.parse_args([*argv, "-f", str(audio)])
    main.process(audio, Config(), out, args)
    return next(out.iterdir())


def test_process_writes_both_transcripts_and_summaries(tmp_path, pipeline):
    run_dir = run_process(tmp_path, [])

    assert run_dir.name.startswith("clip_")
    assert (run_dir / "transcript.txt").read_text(encoding="utf-8") == "hola mundo"
    assert (run_dir / "transcript.en.txt").read_text(encoding="utf-8") == "hello world"
    assert (run_dir / "summary.md").read_text(encoding="utf-8") == "## Summary\nhola mundo"
    assert (run_dir / "summary.en.md").read_text(encoding="utf-8") == "## Summary\nhello world"
    assert pipeline["summarized"] == ["hola mundo", "hello world"]


def test_process_transcribe_only_skips_the_summary(tmp_path, pipeline):
    run_dir = run_process(tmp_path, ["--transcribe-only"])

    assert sorted(p.name for p in run_dir.iterdir()) == ["transcript.en.txt", "transcript.txt"]
    assert pipeline["summarized"] == []


def test_process_omits_the_english_copy_for_english_audio(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(
        transcribe,
        "run",
        lambda path, config, api_key=None: transcribe.Result(text="hello", language="en"),
    )

    run_dir = run_process(tmp_path, [])

    assert sorted(p.name for p in run_dir.iterdir()) == ["summary.md", "transcript.txt"]


def test_main_reports_a_missing_input_file(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "check_system_requirements", lambda config: None)

    assert main.main(["-f", "nope.mp3"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_main_reports_a_failed_download(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "check_system_requirements", lambda config: None)
    monkeypatch.setattr(fetch, "run", lambda *a: (_ for _ in ()).throw(fetch.FetchError("gone")))

    assert main.main(["-u", "https://youtu.be/gone"]) == 1
    assert "gone" in capsys.readouterr().err


def test_main_succeeds_from_a_url(tmp_path, monkeypatch, pipeline):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "check_system_requirements", lambda config: None)

    def fake_fetch(url, dest_dir, config):
        dest_dir.mkdir(parents=True, exist_ok=True)
        clip = dest_dir / "A Talk-abc123.webm"
        clip.touch()
        return clip

    monkeypatch.setattr(fetch, "run", fake_fetch)

    assert main.main(["-u", "https://youtu.be/abc123"]) == 0

    run_dir = next(Path("data/output").iterdir())
    assert run_dir.name.startswith("A Talk-abc123_")
    assert (run_dir / "summary.md").exists()


def test_main_reports_a_broken_config(tmp_path, capsys):
    bad = tmp_path / "config.yaml"
    bad.write_text("transcribe:\n  nonsense: 1\n", encoding="utf-8")

    assert main.main(["-f", "clip.mp3", "--config", str(bad)]) == 1
    assert "unknown key" in capsys.readouterr().err


def test_main_returns_one_when_processing_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clip.mp3").touch()
    monkeypatch.setattr(main, "check_system_requirements", lambda config: None)

    def boom(*args, **kwargs):
        raise RuntimeError("gpu fell over")

    monkeypatch.setattr(main, "process", boom)

    assert main.main(["-f", "clip.mp3"]) == 1
    assert "gpu fell over" in capsys.readouterr().err


def test_main_succeeds_end_to_end(tmp_path, monkeypatch, capsys, pipeline):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clip.mp3").touch()
    monkeypatch.setattr(main, "check_system_requirements", lambda config: None)

    assert main.main(["-f", "clip.mp3"]) == 0

    run_dir = next(Path("data/output").iterdir())
    assert (run_dir / "summary.md").exists()
