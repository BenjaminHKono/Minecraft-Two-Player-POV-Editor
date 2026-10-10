import json

import pytest

from src import media_info
from src.config import OutputConfig, config_from_dict
from src.edit_planner import SpeechSpan, plan_edit
from src.errors import FFmpegNotFoundError, MediaError, RenderError
from src.media_info import parse_probe, probe
from src.renderer import build_command, build_filter_graph, render
from src.timeline import SyncMapping
from tests.conftest import needs_ffmpeg

SOURCES = {"player_a": "a.mp4", "player_b": "b.mp4"}


def _edl():
    m = SyncMapping.from_b_offset(-2.0, 62.0, 60.0)
    spans = [SpeechSpan(2, 8, "S0", "player_a"), SpeechSpan(10, 30, "S1", "player_b")]
    return plan_edit(m, SOURCES, spans, config_from_dict({}).editing)


def test_command_generation():
    edl = _edl()
    out = OutputConfig(resolution="1280x720", fps=30)
    cmd = build_command(edl, out, "out.mp4", "f.txt")
    assert cmd.count("-i") == 2 * len(edl.segments)
    seg = edl.segments[1]
    assert f"{seg.source_start:.6f}" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-c:a") + 1] == "aac"
    graph = build_filter_graph(edl, out)
    assert "scale=1280:720" in graph and "fps=30" in graph and f"concat=n={len(edl.segments)}" in graph
    # one continuous audio track, taken from player_a only
    audio_inputs = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-i"][1::2]
    assert set(audio_inputs) == {"a.mp4"}


def test_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(media_info.shutil, "which", lambda n: None)
    monkeypatch.delenv("FFPROBE_PATH", raising=False)
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    with pytest.raises(FFmpegNotFoundError):
        probe(f)
    with pytest.raises(FFmpegNotFoundError):
        render(_edl_with_files(tmp_path), OutputConfig(), tmp_path / "o.mp4")


def _edl_with_files(tmp_path):
    m = SyncMapping.from_b_offset(0.0, 60.0, 60.0)
    from src.edit_planner import plan_unedited
    return plan_unedited(m, {"player_a": "a.mp4", "player_b": "b.mp4"})


def test_parse_probe_errors():
    with pytest.raises(MediaError, match="no video"):
        parse_probe("x", {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}})
    with pytest.raises(MediaError, match="no audio"):
        parse_probe("x", {"streams": [{"codec_type": "video"}], "format": {"duration": "1"}})
    with pytest.raises(MediaError, match="duration"):
        parse_probe("x", {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {}})


@needs_ffmpeg
def test_probe_missing_and_corrupt(tmp_path):
    with pytest.raises(MediaError, match="not found"):
        probe(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 10)
    with pytest.raises(MediaError):
        probe(bad)
    txt = tmp_path / "a.txt"
    txt.write_text("hello")
    with pytest.raises(MediaError):
        probe(txt)


@needs_ffmpeg
def test_probe_and_invalid_edl_render(media_pair, tmp_path):
    a, b = media_pair
    ia, ib = probe(a), probe(b)
    assert ia.width == 320 and ib.width == 256 and round(ib.fps) == 24
    assert ia.duration == pytest.approx(20, abs=0.2) and ib.duration == pytest.approx(22, abs=0.2)
    edl = _edl_with_files(tmp_path)
    with pytest.raises(Exception):
        render(edl, OutputConfig(), tmp_path / "o.mp4")  # source files do not exist
    assert not (tmp_path / "o.mp4").exists()
