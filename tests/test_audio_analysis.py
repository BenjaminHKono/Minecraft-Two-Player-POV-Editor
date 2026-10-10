import numpy as np
import pytest
import soundfile as sf

from src.audio_analysis import Interval, detect_speech, frame_energies_db, invert_intervals


def make_wav(path, segments, total=10.0, rate=16000):
    x = np.zeros(int(total * rate), dtype=np.float32)
    rng = np.random.default_rng(0)
    for s, e in segments:
        x[int(s * rate):int(e * rate)] = rng.normal(0, 0.1, int(e * rate) - int(s * rate))
    sf.write(str(path), x, rate)


def test_speech_detection_and_padding(tmp_path):
    make_wav(tmp_path / "a.wav", [(2, 3), (6, 7)])
    levels, frame = frame_energies_db(tmp_path / "a.wav")
    plain = detect_speech(levels, frame)
    assert [(round(i.start, 1), round(i.end, 1)) for i in plain] == [(2.0, 3.0), (6.0, 7.0)]
    padded = detect_speech(levels, frame, padding_seconds=0.25)
    assert padded[0].start == pytest.approx(1.75, abs=0.03) and padded[0].end == pytest.approx(3.25, abs=0.03)


def test_short_blips_dropped_and_gaps_merged(tmp_path):
    make_wav(tmp_path / "a.wav", [(1, 1.05), (4, 5), (5.2, 6)])
    levels, frame = frame_energies_db(tmp_path / "a.wav")
    out = detect_speech(levels, frame)
    assert len(out) == 1 and out[0].start == pytest.approx(4.0, abs=0.03) and out[0].end == pytest.approx(6.0, abs=0.03)


def test_invert_intervals():
    gaps = invert_intervals([Interval(2, 3), Interval(6, 7)], 10)
    assert [(g.start, g.end) for g in gaps] == [(0, 2), (3, 6), (7, 10)]
