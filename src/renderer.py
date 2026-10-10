"""FFmpeg renderer: turns an :class:`EDL` into an MP4 with one continuous audio track."""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .config import OutputConfig
from .edit_decision_list import EDL
from .errors import RenderError
from .media_info import find_executable, probe

log = logging.getLogger(__name__)


def build_filter_graph(edl: EDL, out: OutputConfig) -> str:
    """Filter graph: input 2*i is segment i's video, 2*i+1 its audio (both pre-seeked to the segment start)."""
    w, h = out.size
    parts: list[str] = []
    labels: list[str] = []
    for i in range(len(edl.segments)):
        parts.append(
            f"[{2 * i}:v:0]setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={out.fps},format=yuv420p[v{i}]")
        parts.append(f"[{2 * i + 1}:a:0]asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(edl.segments)}:v=1:a=1[vout][aout]")
    return ";\n".join(parts)


def build_command(edl: EDL, out: OutputConfig, output_path: str | Path, filter_graph: str,
                  ffmpeg: str = "ffmpeg") -> list[str]:
    """Build the ffmpeg command line. Each segment is read with its own ``-ss/-t`` input (accurate seek)."""
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-y", "-progress", "pipe:1", "-nostats"]
    for s in edl.segments:
        dur = s.source_end - s.source_start
        cmd += ["-ss", f"{s.source_start:.6f}", "-t", f"{dur:.6f}", "-i", s.source_file]
        a = s.audio
        cmd += ["-ss", f"{a['source_start']:.6f}", "-t", f"{dur:.6f}", "-i", a["source_file"]]
    cmd += ["-filter_complex", filter_graph, "-map", "[vout]", "-map", "[aout]",
            "-c:v", out.video_codec, "-crf", str(out.crf), "-preset", out.preset,
            "-c:a", out.audio_codec, "-b:a", out.audio_bitrate, "-movflags", "+faststart",
            "-f", "mp4", str(output_path)]
    return cmd


def render(edl: EDL, out: OutputConfig, output_path: str | Path,
           progress: Callable[[float], None] | None = None) -> Path:
    """Render ``edl``. Writes to a temporary file first so failures never leave a fake-complete export."""
    edl.validate()
    ffmpeg = find_executable("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
    total = edl.output_duration
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="povedit_") as td:
        cmd = build_command(edl, out, tmp_out, build_filter_graph(edl, out), ffmpeg)
        log.info("Rendering %d segments (%.1fs) to %s", len(edl.segments), total, output_path)
        log.debug("ffmpeg command: %s", cmd)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("out_time_us=") and progress:
                    try:
                        progress(min(1.0, int(line.split("=")[1]) / 1e6 / total))
                    except ValueError:
                        pass
            err = proc.stderr.read() if proc.stderr else ""
            code = proc.wait()
        except BaseException:
            tmp_out.unlink(missing_ok=True)
            raise
    if code != 0:
        tmp_out.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg failed with exit code {code}: {err.strip()[-2000:]}")
    try:
        info = probe(tmp_out)
    except Exception as exc:
        tmp_out.unlink(missing_ok=True)
        raise RenderError(f"Rendered file is not readable: {exc}") from exc
    if abs(info.duration - total) > max(1.0, 0.02 * total):
        tmp_out.unlink(missing_ok=True)
        raise RenderError(f"Rendered duration {info.duration:.2f}s does not match expected {total:.2f}s")
    tmp_out.replace(output_path)
    log.info("Render finished in %.1fs: %s (%.2fs)", time.monotonic() - start, output_path, info.duration)
    return output_path
