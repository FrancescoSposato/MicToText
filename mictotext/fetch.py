"""Download the audio track of an online video (YouTube and ~1800 other sites).

No external FFmpeg is required: an audio-only stream is downloaded as-is, with no
post-processing, and faster-whisper decodes it through PyAV (which bundles FFmpeg).
When a site only offers muxed streams, the video file is downloaded instead and its
audio track is decoded the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
# Audio-only, always. Both selectors match streams with no video track, so a site that
# only serves muxed files fails loudly instead of pulling down a video we would throw
# away: only the audio is ever transcribed, and a long lecture's video is orders of
# magnitude larger than its audio.
_FORMAT = "bestaudio[ext=m4a]/bestaudio"


class FetchError(RuntimeError):
    """Raised when the media cannot be downloaded."""


@dataclass
class FetchedMedia:
    path: Path
    title: str
    duration: float | None
    webpage_url: str


def is_url(value: str) -> bool:
    return bool(_URL_RE.match(value.strip()))


def _progress_hook(status: dict) -> None:
    if status.get("status") != "downloading":
        if status.get("status") == "finished":
            print("\n  Download completato, preparazione del file...", flush=True)
        return
    total = status.get("total_bytes") or status.get("total_bytes_estimate")
    downloaded = status.get("downloaded_bytes", 0)
    if total:
        percent = downloaded / total * 100
        print(f"\r  Download: {percent:5.1f}%  ({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  Download: {downloaded / 1e6:.1f} MB", end="", flush=True)


def download_audio(url: str, dest_dir: Path) -> FetchedMedia:
    """Download the best audio stream of `url` into `dest_dir`. Returns the local file."""
    if not is_url(url):
        raise FetchError(f"Non sembra un URL valido: {url!r}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": _FORMAT,
        "outtmpl": str(dest_dir / "sorgente.%(ext)s"),
        "noplaylist": True,          # a playlist URL grabs only the selected video
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,          # we print our own progress line
        "progress_hooks": [_progress_hook],
        "retries": 3,
        "socket_timeout": 30,
    }

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except DownloadError as exc:
        raise FetchError(_friendly_error(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises a wide range of errors
        raise FetchError(f"Download non riuscito: {exc}") from exc

    if not path.exists():
        # Some extractors report a different extension than the file finally written.
        candidates = sorted(dest_dir.glob("sorgente.*"))
        if not candidates:
            raise FetchError("Download terminato ma nessun file trovato.")
        path = candidates[0]

    _assert_audio_only(path)

    return FetchedMedia(
        path=path,
        title=info.get("title") or path.stem,
        duration=info.get("duration"),
        webpage_url=info.get("webpage_url") or url,
    )


# Cover art is stored as a single-frame image stream; that is not a video track.
_COVER_ART_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}


def _assert_audio_only(path: Path) -> None:
    """Verify no real video track was downloaded. Reads headers only, so it is cheap."""
    try:
        import av
        with av.open(str(path)) as container:
            video = [
                stream for stream in container.streams.video
                if (stream.codec_context.name or "").lower() not in _COVER_ART_CODECS
            ]
    except FetchError:
        raise
    except Exception:  # noqa: BLE001 - a probe failure must not block a valid download
        return

    if video:
        size_mb = path.stat().st_size / 1e6
        path.unlink(missing_ok=True)
        raise FetchError(
            f"Il file scaricato conteneva una traccia video ({video[0].codec_context.name}, "
            f"{size_mb:.1f} MB): scartato. L'app scarica solo audio."
        )


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if "requested format is not available" in lowered or "no video formats found" in lowered:
        return ("Questo sito non offre una traccia audio separata: servirebbe scaricare "
                "l'intero video, cosa che l'app non fa di proposito. Scarica l'audio a "
                "parte e usalo con --audio.")
    if "private" in lowered:
        return "Il video e' privato."
    if "not available" in lowered or "unavailable" in lowered:
        return "Il video non e' disponibile (rimosso o bloccato nel tuo paese)."
    if "sign in" in lowered or "age" in lowered:
        return "Il video richiede l'accesso a un account (età o login)."
    if "unsupported url" in lowered:
        return "URL non supportato da yt-dlp."
    return f"Download non riuscito: {message.strip().splitlines()[-1][:300]}"
