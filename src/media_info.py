"""FFmpeg/FFprobe discovery and media inspection."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from .errors import FFmpegNotFoundError, MediaError

log = logging.getLogger(__name__)


def find_executable(name: str) -> str:
    """Locate ``ffmpeg``/``ffprobe`` via env var (FFMPEG_PATH / FFPROBE_PATH) or PATH."""
    env = os.environ.get(f"{name.upper()}_PATH")
    path = env if env and Path(env).is_file() else shutil.which(name)
    if not path:
        raise FFmpegNotFoundError(
            f"'{name}' was not found. Install FFmpeg and add its 'bin' folder to PATH, "
            f"or set the {name.upper()}_PATH environment variable (see README)."
        )
    return path


@dataclass
class MediaInfo:
    path: str
    duration: float
    start_time: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    audio_stream_index: int
    video_stream_count: int
    audio_stream_count: int
    size_bytes: int
    mtime: float

    def to_dict(self) -> dict:
        return asdict(self)


def _fps(value: str | None) -> float:
    try:
        f = Fraction(value or "0")
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_probe(path: str, data: dict, size_bytes: int = 0, mtime: float = 0.0) -> MediaInfo:
    """Convert raw ffprobe JSON into :class:`MediaInfo`, validating usable streams."""
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if not videos:
        raise MediaError(f"{path}: no video stream found")
    if not audios:
        raise MediaError(f"{path}: no audio stream found (voice chat audio is required)")
    v, a = videos[0], audios[0]
    fmt = data.get("format", {})
    try:
        duration = float(fmt.get("duration") or v.get("duration") or 0)
        start = float(fmt.get("start_time") or 0)
    except (TypeError, ValueError):
        raise MediaError(f"{path}: unreadable duration/start time") from None
    if duration <= 0:
        raise MediaError(f"{path}: could not determine a positive duration (file may be corrupt)")
    return MediaInfo(
        path=str(path), duration=duration, start_time=start,
        width=int(v.get("width", 0)), height=int(v.get("height", 0)),
        fps=_fps(v.get("avg_frame_rate")) or _fps(v.get("r_frame_rate")),
        video_codec=v.get("codec_name", "unknown"), audio_codec=a.get("codec_name", "unknown"),
        audio_sample_rate=int(a.get("sample_rate", 0) or 0), audio_channels=int(a.get("channels", 0) or 0),
        audio_stream_index=int(a.get("index", 0)),
        video_stream_count=len(videos), audio_stream_count=len(audios),
        size_bytes=size_bytes, mtime=mtime,
    )


def probe(path: str | Path) -> MediaInfo:
    """Inspect a media file with FFprobe."""
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"File not found: {p}")
    cmd = [find_executable("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)]
    log.debug("ffprobe: %s", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"FFprobe could not read {p} (corrupt or unsupported file): {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise MediaError(f"FFprobe returned invalid output for {p}") from None
    st = p.stat()
    info = parse_probe(str(p), data, st.st_size, st.st_mtime)
    log.info("%s: %.2fs %dx%d@%.2ffps %s/%s audio %dHz x%d", p.name, info.duration, info.width,
             info.height, info.fps, info.video_codec, info.audio_codec, info.audio_sample_rate, info.audio_channels)
    return info
