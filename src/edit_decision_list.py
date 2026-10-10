"""Edit decision list (EDL): schema, JSON I/O and validation.

All timestamps are seconds. ``output_*`` is the final video's timeline,
``shared_*`` the shared timeline, ``source_*`` the source file's timeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import EDLError
from .timeline import PLAYERS

EDL_VERSION = 1
EPS = 1e-3
ACTIONS = ("retained", "shortened", "removed")


@dataclass
class Segment:
    output_start: float
    output_end: float
    player: str
    source_file: str
    source_start: float
    source_end: float
    shared_start: float
    shared_end: float
    speakers: list[str] = field(default_factory=list)
    activity: float | None = None
    reason: str = ""
    confidence: float | None = None
    action: str = "retained"
    audio: dict[str, Any] = field(default_factory=dict)
    transition: str = "cut"


@dataclass
class EDL:
    sources: dict[str, dict[str, Any]]  # player -> {"path": str, "duration": float, "offset": float}
    segments: list[Segment]
    removed: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = EDL_VERSION

    @property
    def output_duration(self) -> float:
        return self.segments[-1].output_end if self.segments else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def from_dict(cls, data: dict) -> "EDL":
        if not isinstance(data, dict) or data.get("version") != EDL_VERSION:
            raise EDLError(f"Unsupported EDL version {data.get('version') if isinstance(data, dict) else None!r}; expected {EDL_VERSION}")
        try:
            segs = [Segment(**s) for s in data["segments"]]
            return cls(sources=data["sources"], segments=segs, removed=data.get("removed", []),
                       metadata=data.get("metadata", {}), version=data["version"])
        except (KeyError, TypeError) as exc:
            raise EDLError(f"Malformed EDL: {exc}") from None

    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise EDLError(f"EDL file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise EDLError(f"Invalid EDL JSON: {exc}") from None
        edl = cls.from_dict(data)
        edl.validate()
        return edl

    def validate(self) -> None:
        """Raise :class:`EDLError` if the EDL is inconsistent."""
        if not self.segments:
            raise EDLError("EDL has no segments")
        for p in PLAYERS:
            if p not in self.sources:
                raise EDLError(f"EDL is missing source {p!r}")
        prev_end = 0.0
        for i, s in enumerate(self.segments):
            where = f"segment {i}"
            if s.player not in PLAYERS:
                raise EDLError(f"{where}: invalid player {s.player!r}")
            if s.action not in ACTIONS:
                raise EDLError(f"{where}: invalid action {s.action!r}")
            if s.source_file != self.sources[s.player]["path"]:
                raise EDLError(f"{where}: source_file does not match source for {s.player}")
            for a, b, label in ((s.output_start, s.output_end, "output"), (s.source_start, s.source_end, "source"),
                                (s.shared_start, s.shared_end, "shared")):
                if b - a <= 0:
                    raise EDLError(f"{where}: {label} duration must be positive ({a} .. {b})")
            if abs((s.output_end - s.output_start) - (s.source_end - s.source_start)) > EPS:
                raise EDLError(f"{where}: output and source durations differ")
            if s.output_start < prev_end - EPS:
                raise EDLError(f"{where}: segments overlap or are not chronological")
            if s.output_start > prev_end + EPS:
                raise EDLError(f"{where}: gap in output timeline before this segment")
            prev_end = s.output_end
            dur = self.sources[s.player]["duration"]
            if s.source_start < -EPS or s.source_end > dur + EPS:
                raise EDLError(f"{where}: source range {s.source_start:.3f}..{s.source_end:.3f} is outside "
                               f"{s.player} recording (0..{dur:.3f})")
            audio = s.audio
            if audio:
                ap = audio.get("source_player")
                if ap not in PLAYERS:
                    raise EDLError(f"{where}: invalid audio source player {ap!r}")
                adur = self.sources[ap]["duration"]
                if audio["source_start"] < -EPS or audio["source_end"] > adur + EPS:
                    raise EDLError(f"{where}: audio range outside {ap} recording")
