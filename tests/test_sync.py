import numpy as np
import pytest
import soundfile as sf

from src.errors import SyncError
from src.synchronize import compute_envelope, estimate_offset_from_envelopes, synchronize_audio
from tests.conftest import make_voice


def test_envelope_offset_sign(tmp_path):
    voice = make_voice(120, seed=1)
    a, b = voice[: 100 * 16000], voice[5 * 16000: 110 * 16000] * 0.5
    sf.write(str(tmp_path / "a.wav"), a, 16000)
    sf.write(str(tmp_path / "b.wav"), b, 16000)
    r = synchronize_audio(tmp_path / "a.wav", tmp_path / "b.wav")
    assert r.offset_b_seconds == pytest.approx(-5.0, abs=0.03)  # B started 5 s later
    assert r.confidence > 0.5
    r2 = synchronize_audio(str(tmp_path / "b.wav"), str(tmp_path / "a.wav"))
    assert r2.offset_b_seconds == pytest.approx(5.0, abs=0.03)


def test_low_confidence_warns(tmp_path):
    sf.write(tmp_path / "a.wav", make_voice(60, seed=1), 16000)
    sf.write(tmp_path / "b.wav", make_voice(60, seed=2), 16000)
    r = synchronize_audio(tmp_path / "a.wav", tmp_path / "b.wav")
    assert r.confidence < 0.3 and r.warnings


def test_silent_audio_fails(tmp_path):
    sf.write(tmp_path / "s.wav", np.zeros(16000 * 20, dtype=np.float32), 16000)
    env = compute_envelope(tmp_path / "s.wav")
    with pytest.raises(SyncError):
        estimate_offset_from_envelopes(env, env, 100)
