"""Speaker diarization behind a small adapter interface.

Diarization labels (``SPEAKER_00`` ...) are arbitrary and are *never* assumed to
match Player A / Player B; they are mapped explicitly via ``speakers.mapping``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .audio_analysis import Interval
from .errors import ConfigError, OptionalDependencyError

log = logging.getLogger(__name__)
UNKNOWN_LABEL = "UNKNOWN"


@dataclass
class SpeakerSegment:
    start: float
    end: float
    label: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class Diarizer(Protocol):
    def diarize(self, wav_path: str | Path, speech: list[Interval]) -> list[SpeakerSegment]: ...


class NullDiarizer:
    """Fallback: speech is detected but speakers are unknown (label UNKNOWN, confidence 0)."""

    def diarize(self, wav_path, speech):
        return [SpeakerSegment(s.start, s.end, UNKNOWN_LABEL, 0.0) for s in speech]


class SegmentsFileDiarizer:
    """Load manually provided/precomputed segments: ``[{"start":..,"end":..,"label":..,"confidence":..}]``.

    Times are in the analyzed audio's time (same as the VAD output).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def diarize(self, wav_path, speech):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [SpeakerSegment(float(d["start"]), float(d["end"]), str(d["label"]), float(d.get("confidence", 1.0)))
                    for d in data]
        except FileNotFoundError:
            raise ConfigError(f"Speaker segments file not found: {self.path}") from None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid speaker segments file {self.path}: {exc}") from None


class PyannoteDiarizer:
    """pyannote.audio adapter (optional). Needs the HF_TOKEN environment variable and accepted model terms."""

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1"):
        self.model = model

    def diarize(self, wav_path, speech):
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise OptionalDependencyError("pyannote.audio is not installed (pip install pyannote.audio; see README).") from None
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise OptionalDependencyError("Set the HF_TOKEN environment variable to download pyannote models (see README).")
        pipeline = Pipeline.from_pretrained(self.model, use_auth_token=token)
        result = pipeline(str(wav_path))
        return [SpeakerSegment(turn.start, turn.end, label, 1.0)
                for turn, _, label in result.itertracks(yield_label=True)]


def make_diarizer(kind: str, segments_file: str | None = None) -> Diarizer:
    if kind == "none":
        return NullDiarizer()
    if kind == "segments_file":
        return SegmentsFileDiarizer(segments_file or "")
    if kind == "pyannote":
        return PyannoteDiarizer()
    raise ConfigError(f"Unknown diarizer {kind!r}")


def map_speakers(segments: list[SpeakerSegment], mapping: dict[str, str]) -> list[tuple[SpeakerSegment, str | None]]:
    """Pair each segment with its player (``None`` when the label is unmapped)."""
    return [(s, mapping.get(s.label)) for s in segments]
