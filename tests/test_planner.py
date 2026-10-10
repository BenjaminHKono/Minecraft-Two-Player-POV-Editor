import pytest

from src.config import EditingConfig
from src.edit_decision_list import EDL
from src.edit_planner import SpeechSpan, plan_edit, plan_unedited
from src.errors import EDLError
from src.speaker_detection import SpeakerSegment, map_speakers
from src.timeline import SyncMapping

SOURCES = {"player_a": "a.mp4", "player_b": "b.mp4"}
M = SyncMapping.from_b_offset(-2.0, 62.0, 60.0)  # common range 2..62


def span(s, e, player, conf=1.0):
    return SpeechSpan(s, e, "SPK_" + str(player), player, conf)


def players(edl):
    return [(round(s.shared_start, 1), s.player) for s in edl.segments]


def plan(spans, **kw):
    return plan_edit(M, SOURCES, spans, EditingConfig(**kw))


def test_speaker_mapping():
    segs = [SpeakerSegment(0, 1, "SPEAKER_00"), SpeakerSegment(1, 2, "SPEAKER_01")]
    out = map_speakers(segs, {"SPEAKER_01": "player_b"})
    assert [p for _, p in out] == [None, "player_b"]


def test_alternating_speakers():
    edl = plan([span(2, 10, "player_a"), span(12, 20, "player_b"), span(25, 35, "player_a")])
    assert players(edl) == [(2.0, "player_a"), (12.0, "player_b"), (25.0, "player_a")]
    assert all(s.reason for s in edl.segments)


def test_rapid_switching_prevented():
    spans = [span(2, 10, "player_a"), span(10, 10.5, "player_b"), span(10.5, 14, "player_a"),
             span(14, 14.4, "player_b"), span(14.4, 20, "player_a")]
    edl = plan(spans)
    assert [s.player for s in edl.segments] == ["player_a"]  # blips shorter than 1 s ignored


def test_minimum_shot_and_cooldown():
    spans = [span(2, 3.5, "player_a"), span(3.5, 5, "player_b"), span(5, 6.5, "player_a"), span(6.5, 8, "player_b")]
    edl = plan(spans, default_player="player_a")
    for s in edl.segments[:-1]:
        assert s.shared_end - s.shared_start >= 2.0 - 1e-6
    starts = [s.shared_start for s in edl.segments]
    assert all(b - a >= 3.0 - 1e-6 for a, b in zip(starts, starts[1:]))


def test_overlapping_speech_keeps_current():
    edl = plan([span(2, 10, "player_b"), span(12, 30, "player_a"), span(12, 30, "player_b")])
    assert players(edl) == [(2.0, "player_b")]


def test_low_confidence_ignored():
    edl = plan([span(2, 20, "player_b", conf=0.2)])
    assert [s.player for s in edl.segments] == ["player_a"]


def test_unmapped_speakers_ignored():
    edl = plan([SpeechSpan(2, 20, "SPEAKER_00", None, 1.0)])
    assert len(edl.segments) == 1


def test_activity_switch_when_silent_and_disabled():
    act = lambda p, a, b: 0.9 if p == "player_b" else 0.1
    on = plan_edit(M, SOURCES, [], EditingConfig(), activity=act)
    off = plan_edit(M, SOURCES, [], EditingConfig(use_gameplay_activity=False), activity=act)
    assert on.segments[-1].player == "player_b" and "more active" in on.segments[-1].reason
    assert len(off.segments) == 1


def test_silent_gameplay_preserved():
    # long silence: nothing is removed, output covers the full common range
    edl = plan([span(2, 4, "player_a"), span(50, 60, "player_a")])
    assert edl.output_duration == pytest.approx(60.0)
    assert all(s.action == "retained" for s in edl.segments)


def test_edl_timestamps_use_offsets():
    edl = plan([span(2, 8, "player_a"), span(10, 30, "player_b")])
    seg = edl.segments[-1]
    assert seg.player == "player_b"
    assert seg.source_start == pytest.approx(seg.shared_start - 2.0)
    assert seg.audio["source_player"] == "player_a"
    assert seg.audio["source_start"] == pytest.approx(seg.shared_start)
    assert seg.output_start == pytest.approx(seg.shared_start - 2.0)
    edl.validate()


def test_edl_roundtrip_and_validation(tmp_path):
    edl = plan_unedited(M, SOURCES)
    p = tmp_path / "e.json"
    edl.save(p)
    loaded = EDL.load(p)
    assert loaded.version == 1 and len(loaded.segments) == 1
    d = loaded.to_dict()

    def broken(mut):
        e = EDL.from_dict(__import__("copy").deepcopy(d))
        mut(e)
        with pytest.raises(EDLError):
            e.validate()

    broken(lambda e: setattr(e.segments[0], "source_end", 1e6))
    broken(lambda e: setattr(e.segments[0], "source_start", -5.0))
    broken(lambda e: setattr(e.segments[0], "output_end", e.segments[0].output_start))
    broken(lambda e: setattr(e.segments[0], "player", "player_z"))
    broken(lambda e: setattr(e.segments[0], "source_file", "other.mp4"))
    broken(lambda e: e.segments.clear())
    with pytest.raises(EDLError):
        EDL.from_dict({**d, "version": 99})


def test_edl_ordering_and_gaps():
    edl = plan([span(2, 8, "player_a"), span(10, 30, "player_b")])
    assert len(edl.segments) == 2
    edl.segments[1].output_start += 1
    with pytest.raises(EDLError):
        edl.validate()
    edl.segments[1].output_start -= 5
    with pytest.raises(EDLError):
        edl.validate()


def test_initial_camera_follows_first_speaker():
    edl = plan([span(10, 30, "player_b")])
    assert [s.player for s in edl.segments] == ["player_b"]
