import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not available")


def make_voice(seconds: float, rate: int = 16000, seed: int = 0) -> np.ndarray:
    """Synthetic 'voice chat': noise bursts of varying length separated by silence."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * rate), dtype=np.float32)
    t = 0.0
    while t < seconds - 1:
        dur = rng.uniform(0.3, 1.5)
        end = min(seconds, t + dur)
        out[int(t * rate):int(end * rate)] = rng.normal(0, rng.uniform(0.1, 0.3), int(end * rate) - int(t * rate))
        t = end + rng.uniform(0.2, 1.0)
    return out


def make_video(path, voice: np.ndarray, rate: int = 16000, size="320x240", fps=30, tmp=None):
    wav = path.with_suffix(".wav")
    sf.write(str(wav), voice, rate)
    dur = len(voice) / rate
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={dur}",
                    "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                   check=True)
    wav.unlink()
    return path


@pytest.fixture
def media_pair(tmp_path):
    """Player B started 2.0 s after player A and has a different size/fps/duration."""
    voice = make_voice(24, seed=3)
    a = make_video(tmp_path / "a.mp4", voice[: 20 * 16000])
    b = make_video(tmp_path / "b.mp4", voice[2 * 16000: 24 * 16000] * 0.6, size="256x144", fps=24)
    return a, b
