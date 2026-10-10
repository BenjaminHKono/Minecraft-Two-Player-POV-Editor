"""Shared-audio analysis: energy-based voice activity detection and optional transcription.

Timestamps returned here are in the *analyzed file's* time; the pipeline converts
them to the shared timeline.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import OptionalDependencyError

log = logging.getLogger(__name__)


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_energies_db(wav_path: str | Path, frame_seconds: float = 0.02, block_seconds: float = 60.0) -> tuple[np.ndarray, float]:
    """Per-frame RMS level in dBFS, computed block-wise. Returns (levels, frame_seconds)."""
    levels: list[np.ndarray] = []
    with sf.SoundFile(str(wav_path)) as f:
        hop = max(1, int(f.samplerate * frame_seconds))
        step = hop * max(1, int(block_seconds / frame_seconds))
        carry = np.zeros(0, dtype=np.float32)
        for chunk in f.blocks(blocksize=step, dtype="float32", always_2d=False):
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            data = np.concatenate([carry, chunk])
            n = len(data) // hop
            if n:
                rms = np.sqrt((data[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
                levels.append(20 * np.log10(rms + 1e-9))
            carry = data[n * hop:]
        real_frame = hop / f.samplerate
    return (np.concatenate(levels) if levels else np.zeros(0)), real_frame


def detect_speech(levels_db: np.ndarray, frame_seconds: float, threshold_db: float = -40.0,
                  min_speech_seconds: float = 0.2, merge_gap_seconds: float = 0.3,
                  padding_seconds: float = 0.0) -> list[Interval]:
    """Threshold frame levels into speech intervals; merge small gaps, drop tiny blips.

    Each kept interval is extended by ``padding_seconds`` on both sides (clipped to the audio)."""
    active = levels_db > threshold_db
    raw: list[Interval] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            raw.append(Interval(start * frame_seconds, i * frame_seconds))
            start = None
    if start is not None:
        raw.append(Interval(start * frame_seconds, len(active) * frame_seconds))
    merged: list[Interval] = []
    for iv in raw:
        if merged and iv.start - merged[-1].end <= merge_gap_seconds:
            merged[-1].end = iv.end
        else:
            merged.append(Interval(iv.start, iv.end))
    kept = [iv for iv in merged if iv.duration >= min_speech_seconds]
    total = len(levels_db) * frame_seconds
    padded: list[Interval] = []
    for iv in kept:
        lo, hi = max(0.0, iv.start - padding_seconds), min(total, iv.end + padding_seconds)
        if padded and lo <= padded[-1].end:
            padded[-1].end = max(padded[-1].end, hi)
        else:
            padded.append(Interval(lo, hi))
    return padded


def invert_intervals(speech: list[Interval], total: float) -> list[Interval]:
    """Non-speech intervals in ``[0, total]``."""
    out, cur = [], 0.0
    for iv in speech:
        if iv.start > cur:
            out.append(Interval(cur, iv.start))
        cur = max(cur, iv.end)
    if cur < total:
        out.append(Interval(cur, total))
    return out


def transcribe(wav_path: str | Path, model_size: str = "small", language: str | None = None) -> list[dict]:
    """Transcribe with faster-whisper (optional dependency). Returns segment dicts with word timestamps."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise OptionalDependencyError(
            "faster-whisper is not installed. Run: pip install faster-whisper (see README)."
        ) from None
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(wav_path), language=language, word_timestamps=True, vad_filter=True)
    out = []
    for s in segments:
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                    "confidence": float(np.exp(s.avg_logprob)),
                    "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                              for w in (s.words or [])]})
    return out


def intervals_to_dicts(ivs: list[Interval]) -> list[dict]:
    return [asdict(i) for i in ivs]
