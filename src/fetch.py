"""Turn a URL into a local media file under the input directory.
"""

import re
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.config import FetchConfig

# Only these two. A URL is user input, and urlopen speaks file:// and ftp:// too.
SCHEMES = ("http", "https")

# Suffixes that mark a link as a plain file to pull down rather than a page to scrape.
MEDIA_SUFFIXES = {
    ".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga",
    ".oga", ".ogg", ".opus", ".wav", ".webm",
}

# Anything outside this is replaced. The stem also becomes the name of the output
# run directory, so it has to survive both filesystems and a shell.
UNSAFE = re.compile(r"[^\w.\- ]+")
MAX_STEM = 80

# Plain urllib announces itself as Python-urllib, which some CDNs turn away.
USER_AGENT = "Mozilla/5.0 (compatible; briefly/0.2)"


class FetchError(RuntimeError):
    """A URL could not be turned into a local file."""


@dataclass
class Source:
    name: str
    handles: Callable[[str], bool]
    fetch: Callable[[str, Path, FetchConfig], Path]


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def safe_stem(name: str) -> str:
    stem = UNSAFE.sub("_", name).strip(" ._-")
    return stem[:MAX_STEM].strip(" ._-") or "download"


def _is_direct_media(url: str) -> bool:
    return Path(urlparse(url).path).suffix.lower() in MEDIA_SUFFIXES


def _fetch_direct(url: str, dest_dir: Path, config: FetchConfig) -> Path:
    name = Path(unquote(urlparse(url).path))
    target = dest_dir / f"{safe_stem(name.stem)}{name.suffix.lower()}"
    if target.exists():
        print(f"  already downloaded: {target}")
        return target

    # Write beside the target and rename at the end, so an interrupted download
    # never leaves a truncated file that the next run would happily reuse.
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=config.timeout_seconds) as response,
            partial.open("wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {exc}") from exc

    partial.replace(target)
    return target


def _load_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - yt-dlp is a hard dependency
        raise FetchError(
            "yt-dlp is not installed, so links cannot be downloaded. Run `uv sync`."
        ) from exc
    return yt_dlp


def _ytdlp_options(dest_dir: Path, config: FetchConfig) -> dict:
    options = {
        "format": config.format,
        "outtmpl": {"default": str(dest_dir / config.filename_template)},
        "noplaylist": config.noplaylist,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": config.timeout_seconds,
    }
    if config.cookies_file:
        options["cookiefile"] = config.cookies_file
    return options


def _fetch_ytdlp(url: str, dest_dir: Path, config: FetchConfig) -> Path:
    yt_dlp = _load_yt_dlp()

    with yt_dlp.YoutubeDL(_ytdlp_options(dest_dir, config)) as ydl:
        try:
            # Metadata first: it names the file, which tells us whether the
            # download can be skipped, and it catches a playlist before we pull
            # down every item in it.
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise FetchError(f"could not read {url}: {exc}") from exc

        if info.get("_type") == "playlist":
            raise FetchError(
                f"{url} is a playlist, and briefly processes one file per run. "
                f"Pass the URL of a single video."
            )

        target = Path(ydl.prepare_filename(info))
        if target.exists():
            print(f"  already downloaded: {target}")
            return target

        title = info.get("title", url)
        duration = info.get("duration")
        length = f", {duration // 60}m{duration % 60:02d}s" if duration else ""
        print(f"  {title}{length}")

        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise FetchError(f"could not download {url}: {exc}") from exc

    # The container the format selector settled on can differ from the one
    # prepare_filename guessed, so trust what yt-dlp says it actually wrote.
    downloads = info.get("requested_downloads") or []
    written = next((d["filepath"] for d in downloads if d.get("filepath")), None)
    if written is None:
        raise FetchError(f"yt-dlp reported no downloaded file for {url}")
    return Path(written)


SOURCES = [
    Source("direct link", _is_direct_media, _fetch_direct),
    Source("yt-dlp", lambda url: True, _fetch_ytdlp),
]


def run(url: str, dest_dir: Path, config: FetchConfig) -> Path:
    """Download `url` into `dest_dir` and return the file it landed in."""
    if urlparse(url).scheme not in SCHEMES:
        raise FetchError(f"only http and https links are supported: {url}")

    source = next((s for s in SOURCES if s.handles(url)), None)
    if source is None:  # pragma: no cover - the last source claims everything
        raise FetchError(f"no source knows how to handle {url}")

    print(f"Fetching {url} ({source.name}) ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = source.fetch(url, dest_dir, config)
    if not path.exists():
        raise FetchError(f"{source.name} reported {path}, but it is not there")
    return path
