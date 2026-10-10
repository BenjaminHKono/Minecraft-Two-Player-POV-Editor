import json

import pytest

from src import pipeline
from src.cli import main
from src.errors import RenderError
from src.media_info import probe
from tests.conftest import needs_ffmpeg


@needs_ffmpeg
def test_inspect_sync_and_run(media_pair, tmp_path, capsys):
    a, b = media_pair
    work, out = tmp_path / "work", tmp_path / "final.mp4"
    segs = tmp_path / "segs.json"
    segs.write_text(json.dumps([{"start": 1, "end": 9, "label": "SPEAKER_00"}, {"start": 10, "end": 18, "label": "SPEAKER_01"}]))
    base = ["--player-a", str(a), "--player-b", str(b), "--work-dir", str(work)]
    assert main(["inspect", *base]) == 0
    assert main(["synchronize", *base]) == 0
    printed = capsys.readouterr().out
    assert "-2.0" in printed or "-1.9" in printed
    assert main(["run", *base, "--output", str(out), "--speaker-segments", str(segs),
                 "--map", "SPEAKER_00=player_a", "--map", "SPEAKER_01=player_b",
                 "--offset", "-2.0"]) == 0
    info = probe(out)
    assert info.duration == pytest.approx(18.0, abs=0.5)  # common range: shared 2..20
    assert (info.width, info.height) == (1920, 1080)
    edl = json.loads((work / "edl.json").read_text())
    assert [s["player"] for s in edl["segments"]] == ["player_a", "player_b"]
    assert main(["validate", str(work / "edl.json")]) == 0
    assert not list(tmp_path.glob("*.part*"))


@needs_ffmpeg
def test_cli_errors(tmp_path):
    assert main(["inspect", "--player-a", str(tmp_path / "x.mp4"), "--player-b", str(tmp_path / "y.mp4")]) == 1
    assert main(["validate", str(tmp_path / "missing.json")]) == 1


@needs_ffmpeg
def test_stale_cache_invalidated(media_pair, tmp_path):
    a, b = media_pair
    proj = pipeline.load_project(None, {"player_a": str(a), "player_b": str(b), "work_dir": str(tmp_path / "w")})
    s1 = pipeline.synchronize(proj)
    proj.config.synchronization.max_offset_seconds = 100.0
    key_before = json.loads((tmp_path / "w" / "sync.json").read_text())["key"]
    pipeline.synchronize(proj)
    assert json.loads((tmp_path / "w" / "sync.json").read_text())["key"] != key_before
    assert s1.offset_b_seconds == pytest.approx(-2.0, abs=0.05)
