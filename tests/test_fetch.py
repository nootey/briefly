import io
import re
import urllib.error
from pathlib import Path
from typing import ClassVar

import pytest

from src import fetch
from src.config import FetchConfig


@pytest.mark.parametrize(
    "source, expected",
    [
        ("https://youtu.be/abc", True),
        ("http://example.com/clip.mp3", True),
        ("clip.mp3", False),
        ("data/input/clip.mp3", False),
        ("ftp://example.com/clip.mp3", False),
    ],
)
def test_is_url(source, expected):
    assert fetch.is_url(source) is expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("a normal title", "a normal title"),
        ("slashes/and:colons", "slashes_and_colons"),
        ("...", "download"),
        ("", "download"),
        ("x" * 200, "x" * fetch.MAX_STEM),
    ],
)
def test_safe_stem(name, expected):
    assert fetch.safe_stem(name) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com/clip.mp3", True),
        ("https://example.com/clip.WEBM", True),
        ("https://example.com/clip.mp3?token=1", True),
        ("https://example.com/clip.txt", False),
        ("https://www.youtube.com/watch?v=abc", False),
    ],
)
def test_direct_media_claims_only_media_links(url, expected):
    assert fetch._is_direct_media(url) is expected


def test_run_rejects_a_scheme_it_does_not_speak(tmp_path):
    with pytest.raises(fetch.FetchError, match="only http and https"):
        fetch.run("file:///etc/passwd", tmp_path, FetchConfig())


def test_run_picks_the_first_source_that_claims_the_url(tmp_path, monkeypatch):
    called = []

    def record(name):
        def fetcher(url, dest_dir, config):
            called.append(name)
            target = dest_dir / "clip.mp3"
            target.touch()
            return target

        return fetcher

    monkeypatch.setattr(
        fetch,
        "SOURCES",
        [
            fetch.Source("direct link", fetch._is_direct_media, record("direct")),
            fetch.Source("yt-dlp", lambda url: True, record("ytdlp")),
        ],
    )

    fetch.run("https://example.com/clip.mp3", tmp_path, FetchConfig())
    fetch.run("https://youtu.be/abc", tmp_path, FetchConfig())

    assert called == ["direct", "ytdlp"]


def test_run_complains_when_a_source_lies_about_the_file(tmp_path, monkeypatch):
    ghost = fetch.Source("ghost", lambda url: True, lambda u, d, c: d / "nope.mp3")
    monkeypatch.setattr(fetch, "SOURCES", [ghost])

    with pytest.raises(fetch.FetchError, match="not there"):
        fetch.run("https://example.com/clip.mp3", tmp_path, FetchConfig())


def fake_urlopen(payload=b"audio", error=None):
    def opener(request, timeout=None):
        if error:
            raise error
        return io.BytesIO(payload)

    return opener


def test_fetch_direct_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen(b"audio"))

    path = fetch._fetch_direct("https://example.com/My%20Clip.MP3", tmp_path, FetchConfig())

    assert path == tmp_path / "My Clip.mp3"
    assert path.read_bytes() == b"audio"


def test_fetch_direct_reuses_an_existing_file(tmp_path, monkeypatch):
    (tmp_path / "clip.mp3").write_bytes(b"old")

    def explode(*args, **kwargs):
        raise AssertionError("should not have downloaded anything")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", explode)

    path = fetch._fetch_direct("https://example.com/clip.mp3", tmp_path, FetchConfig())

    assert path.read_bytes() == b"old"


def test_fetch_direct_leaves_nothing_behind_when_it_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch.urllib.request,
        "urlopen",
        fake_urlopen(error=urllib.error.URLError("no route to host")),
    )

    with pytest.raises(fetch.FetchError, match="could not download"):
        fetch._fetch_direct("https://example.com/clip.mp3", tmp_path, FetchConfig())

    assert list(tmp_path.iterdir()) == []


class FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL, recording what it was asked to do."""

    instances: ClassVar[list] = []

    def __init__(self, options, info=None, downloads=None, raises=None):
        self.options = options
        self.info = info or {"title": "A Talk", "id": "abc123", "ext": "webm", "duration": 125}
        self.downloads = downloads
        self.raises = raises
        self.downloaded = []
        FakeYoutubeDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if self.raises:
            raise self.raises
        if download:
            self.downloaded.append(url)
            return {**self.info, "requested_downloads": self.downloads}
        return self.info

    def prepare_filename(self, info):
        # Enough of yt-dlp's output template to cover the default, including the
        # `%(title).80s` length cap.
        def render(match):
            field, width = match.group(1), match.group(2)
            value = str(info.get(field, ""))
            return value[: int(width)] if width else value

        return re.sub(
            r"%\((\w+)\)(?:\.(\d+))?s", render, self.options["outtmpl"]["default"]
        )


class FakeDownloadError(Exception):
    pass


@pytest.fixture
def yt_dlp(monkeypatch):
    """Install a fake yt_dlp module and hand back a knob for its behaviour."""
    FakeYoutubeDL.instances = []
    settings = {}

    class Module:
        utils = type("utils", (), {"DownloadError": FakeDownloadError})

        @staticmethod
        def YoutubeDL(options):  # mirrors the real class name
            return FakeYoutubeDL(options, **settings)

    monkeypatch.setattr(fetch, "_load_yt_dlp", lambda: Module)
    return settings


def test_fetch_ytdlp_returns_the_path_yt_dlp_reports(tmp_path, yt_dlp, capsys):
    written = tmp_path / "A_Talk-abc123.opus"
    written.touch()
    yt_dlp["downloads"] = [{"filepath": str(written)}]

    path = fetch._fetch_ytdlp("https://youtu.be/abc123", tmp_path, FetchConfig())

    # The reported path wins over the guess, because the format selector can
    # settle on a different container than prepare_filename assumed.
    assert path == written
    assert "A Talk, 2m05s" in capsys.readouterr().out
    assert FakeYoutubeDL.instances[0].downloaded == ["https://youtu.be/abc123"]


def test_fetch_ytdlp_skips_the_download_when_the_file_is_there(tmp_path, yt_dlp):
    (tmp_path / "A Talk-abc123.webm").touch()

    path = fetch._fetch_ytdlp("https://youtu.be/abc123", tmp_path, FetchConfig())

    assert path == tmp_path / "A Talk-abc123.webm"
    assert FakeYoutubeDL.instances[0].downloaded == []


def test_fetch_ytdlp_caps_a_long_title_without_escaping_the_directory(tmp_path, yt_dlp):
    """The title is capped in the template on purpose.

    yt-dlp's trim_file_name option slices the whole rendered path, directory and
    all, so a long title used to truncate the parent directory out of existence
    and write the file somewhere else entirely.
    """
    yt_dlp["info"] = {"title": "T" * 300, "id": "abc123", "ext": "webm"}
    yt_dlp["downloads"] = [{"filepath": str(tmp_path / "capped.webm")}]
    (tmp_path / "capped.webm").touch()

    path = fetch._fetch_ytdlp("https://youtu.be/abc123", tmp_path, FetchConfig())

    assert path.parent == tmp_path
    guess = Path(FakeYoutubeDL.instances[0].prepare_filename(yt_dlp["info"]))
    assert guess.parent == tmp_path
    assert len(guess.name) < 120


def test_fetch_ytdlp_rejects_a_playlist(tmp_path, yt_dlp):
    yt_dlp["info"] = {"_type": "playlist", "title": "Everything", "id": "PL1", "ext": "webm"}

    with pytest.raises(fetch.FetchError, match="playlist"):
        fetch._fetch_ytdlp("https://youtube.com/playlist?list=PL1", tmp_path, FetchConfig())


def test_fetch_ytdlp_reports_an_extractor_failure(tmp_path, yt_dlp):
    yt_dlp["raises"] = FakeDownloadError("Video unavailable")

    with pytest.raises(fetch.FetchError, match="Video unavailable"):
        fetch._fetch_ytdlp("https://youtu.be/gone", tmp_path, FetchConfig())


def test_fetch_ytdlp_notices_an_empty_download(tmp_path, yt_dlp):
    yt_dlp["downloads"] = []

    with pytest.raises(fetch.FetchError, match="no downloaded file"):
        fetch._fetch_ytdlp("https://youtu.be/abc123", tmp_path, FetchConfig())


def test_ytdlp_options_carry_the_config(tmp_path):
    config = FetchConfig(format="worstaudio", cookies_file="cookies.txt", noplaylist=False)

    options = fetch._ytdlp_options(tmp_path, config)

    assert options["format"] == "worstaudio"
    assert options["cookiefile"] == "cookies.txt"
    assert options["noplaylist"] is False
    assert options["outtmpl"]["default"].startswith(str(tmp_path))


def test_ytdlp_options_omit_cookies_when_none_are_configured(tmp_path):
    assert "cookiefile" not in fetch._ytdlp_options(tmp_path, FetchConfig())
