"""Audio-based synchronization via envelope cross-correlation.

Sign convention: the returned ``offset_b_seconds`` satisfies
``source_time_B = shared_time + offset_b_seconds`` with the shared timeline
equal to Player A's timeline (see :mod:`src.timeline`).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlation_lags, fftconvolve

from .errors import SyncError

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    offset_b_seconds: float
    confidence: float
    method: str
    manual: bool = False
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_envelope(wav_path: str | Path, rate_hz: int = 100, block_seconds: float = 60.0) -> np.ndarray:
    """Log-RMS loudness envelope at ``rate_hz`` frames/s, computed block-wise (low memory)."""
    with sf.SoundFile(str(wav_path)) as f:
        hop = max(1, f.samplerate // rate_hz)
        block = hop * max(1, int(block_seconds * rate_hz))
        out: list[np.ndarray] = []
        carry = np.zeros(0, dtype=np.float32)
        for chunk in f.blocks(blocksize=block, dtype="float32", always_2d=False):
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            data = np.concatenate([carry, chunk])
            n = len(data) // hop
            if n:
                frames = data[: n * hop].reshape(n, hop)
                out.append(np.sqrt((frames ** 2).mean(axis=1)))
            carry = data[n * hop:]
    if not out:
        raise SyncError(f"{wav_path}: audio is empty")
    return np.log10(np.concatenate(out) + 1e-5)


def _normalize(env: np.ndarray, rate_hz: int) -> np.ndarray:
    """Remove slow level differences (volume, background sounds) and scale to unit variance."""
    env = env - uniform_filter1d(env, size=max(3, 2 * rate_hz), mode="nearest")
    std = env.std()
    if std < 1e-6:
        raise SyncError("Audio is silent or constant; cannot synchronize")
    return (env - env.mean()) / std


def estimate_offset_from_envelopes(env_a: np.ndarray, env_b: np.ndarray, rate_hz: int,
                                   max_offset_seconds: float = 600.0) -> SyncResult:
    """Estimate B's offset from two loudness envelopes (see module docstring for sign)."""
    a, b = _normalize(env_a, rate_hz), _normalize(env_b, rate_hz)
    corr = fftconvolve(a, b[::-1], mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")
    overlap = (np.minimum(len(b), len(a) - lags) - np.maximum(0, -lags)).clip(1)
    min_overlap = max(rate_hz * 5, int(0.2 * min(len(a), len(b))))
    max_lag = int(max_offset_seconds * rate_hz)
    valid = (overlap >= min(min_overlap, min(len(a), len(b)))) & (np.abs(lags) <= max_lag)
    if not valid.any():
        raise SyncError("Recordings are too short or offset limit too small for synchronization")
    score = np.where(valid, corr / overlap, -np.inf)
    best = int(np.argmax(score))
    peak = float(score[best])
    lag = int(lags[best])
    guard = int(0.5 * rate_hz)
    side = np.where(valid & (np.abs(lags - lag) > guard), score, -np.inf)
    second = float(side.max()) if np.isfinite(side).any() else 0.0
    second = max(second, 0.0)
    if peak <= 0:
        confidence = 0.0
    else:
        distinct = 1.0 - second / peak
        strength = min(1.0, peak / 0.3)
        confidence = float(np.clip(distinct * strength, 0.0, 1.0))
    # a[n + lag] ~ b[n]  =>  b_time = shared_time - lag
    offset = -lag / rate_hz
    return SyncResult(
        offset_b_seconds=offset, confidence=confidence, method="audio_cross_correlation",
        diagnostics={"peak_correlation": peak, "second_peak_correlation": second,
                     "lag_frames": lag, "envelope_rate_hz": rate_hz},
    )


def synchronize_audio(wav_a: str | Path, wav_b: str | Path, rate_hz: int = 100,
                      max_offset_seconds: float = 600.0, min_confidence: float = 0.3) -> SyncResult:
    """Estimate the offset between two extracted WAV files."""
    env_a, env_b = compute_envelope(wav_a, rate_hz), compute_envelope(wav_b, rate_hz)
    result = estimate_offset_from_envelopes(env_a, env_b, rate_hz, max_offset_seconds)
    if result.confidence < min_confidence:
        msg = (f"Low synchronization confidence ({result.confidence:.2f} < {min_confidence}); "
               f"estimated offset {result.offset_b_seconds:+.3f}s may be wrong. "
               f"Verify it and use --offset to correct it.")
        result.warnings.append(msg)
        log.warning(msg)
    log.info("Sync offset_b=%+.3fs confidence=%.2f", result.offset_b_seconds, result.confidence)
    return result


def manual_result(offset: float) -> SyncResult:
    return SyncResult(offset_b_seconds=offset, confidence=1.0, method="manual", manual=True)


def check_drift(wav_a: str | Path, wav_b: str | Path, offset_b: float, rate_hz: int = 100,
                segments: int = 3, tolerance_seconds: float = 0.1) -> list[str]:
    """Re-estimate the offset on parts of the recordings and warn if it changes (drift)."""
    env_a, env_b = compute_envelope(wav_a, rate_hz), compute_envelope(wav_b, rate_hz)
    warnings: list[str] = []
    n = min(len(env_a), len(env_b))
    seg = n // segments
    if seg < rate_hz * 30:
        return warnings
    for i in range(segments):
        sl = slice(i * seg, (i + 1) * seg)
        try:
            r = estimate_offset_from_envelopes(env_a[sl], env_b[sl], rate_hz, max_offset_seconds=abs(offset_b) + 5.0)
        except SyncError:
            continue
        # segment-local offsets are relative to segment start; both start at same index so offsets compare directly
        if r.confidence > 0.3 and abs(r.offset_b_seconds - offset_b) > tolerance_seconds:
            warnings.append(f"Possible drift: segment {i + 1}/{segments} suggests offset {r.offset_b_seconds:+.3f}s "
                            f"vs global {offset_b:+.3f}s")
    return warnings
