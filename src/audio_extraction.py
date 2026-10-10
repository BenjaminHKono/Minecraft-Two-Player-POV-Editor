"""Extract audio from recordings into a consistent analysis format (mono WAV)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .errors import MediaError
from .media_info import MediaInfo, find_executable

log = logging.getLogger(__name__)
ANALYSIS_RATE = 16000


def extract_audio(info: MediaInfo, dest: str | Path, sample_rate: int = ANALYSIS_RATE) -> Path:
    """Write the first audio stream of ``info.path`` as mono 16-bit PCM WAV.

    The WAV is written to a temporary name and renamed, so an interrupted run
    never leaves a partial file that looks complete. Timestamps are relative to
    the start of the audio stream.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    cmd = [find_executable("ffmpeg"), "-v", "error", "-y", "-i", info.path,
           "-map", f"0:{info.audio_stream_index}", "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    log.info("Extracting audio from %s", info.path)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"Audio extraction failed for {info.path}: {proc.stderr.strip()}")
    tmp.replace(dest)
    return dest
