"""Validated JSON configuration.

All values have documented defaults (see README). Unknown keys are rejected so
typos are not silently ignored.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError

PLAYERS = ("player_a", "player_b")


@dataclass
class OutputConfig:
    resolution: str = "1920x1080"
    fps: int = 60
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"
    audio_source: str = "player_a"  # which recording supplies the single voice-chat track

    @property
    def size(self) -> tuple[int, int]:
        w, h = self.resolution.lower().split("x")
        return int(w), int(h)


@dataclass
class EditingConfig:
    minimum_shot_seconds: float = 2.0
    speaker_switch_threshold_seconds: float = 1.0
    speaker_switch_cooldown_seconds: float = 3.0
    speaker_confidence_threshold: float = 0.5
    default_player: str = "player_a"
    remove_silence: bool = True  # shorten long pauses between speech
    use_gameplay_activity: bool = True  # only used when activity scores are supplied
    preserve_uncertain_gameplay: bool = True
    activity_switch_margin: float = 0.3
    planning_step_seconds: float = 0.1


@dataclass
class SilenceConfig:
    short_pause_seconds: float = 1.0
    long_pause_seconds: float = 3.0
    minimum_removal_seconds: float = 5.0
    speech_padding_seconds: float = 0.25
    threshold_db: float = -40.0  # energy VAD threshold relative to full scale
    minimum_speech_seconds: float = 0.2


@dataclass
class SynchronizationConfig:
    method: str = "audio_cross_correlation"
    manual_offset_seconds: float | None = None
    max_offset_seconds: float = 600.0
    min_confidence: float = 0.3
    envelope_rate_hz: int = 100


@dataclass
class SpeakersConfig:
    diarizer: str = "none"  # none | segments_file | pyannote
    segments_file: str | None = None
    mapping: dict[str, str] = field(default_factory=dict)  # diarization label -> player_a/player_b


@dataclass
class Config:
    output: OutputConfig = field(default_factory=OutputConfig)
    editing: EditingConfig = field(default_factory=EditingConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        o, e, s, y, sp = self.output, self.editing, self.silence, self.synchronization, self.speakers
        if not re.fullmatch(r"\d+x\d+", o.resolution) or min(o.size) <= 0 or any(v % 2 for v in o.size):
            raise ConfigError(f"output.resolution must be WIDTHxHEIGHT with even values, got {o.resolution!r}")
        _positive(o.fps, "output.fps")
        if not 0 <= o.crf <= 51:
            raise ConfigError("output.crf must be between 0 and 51")
        _choice(o.audio_source, PLAYERS, "output.audio_source")
        _choice(e.default_player, PLAYERS, "editing.default_player")
        _positive(e.minimum_shot_seconds, "editing.minimum_shot_seconds")
        _positive(e.planning_step_seconds, "editing.planning_step_seconds")
        for name in ("speaker_switch_threshold_seconds", "speaker_switch_cooldown_seconds"):
            _non_negative(getattr(e, name), f"editing.{name}")
        if not 0 <= e.speaker_confidence_threshold <= 1:
            raise ConfigError("editing.speaker_confidence_threshold must be in [0, 1]")
        for name in ("short_pause_seconds", "long_pause_seconds", "minimum_removal_seconds", "speech_padding_seconds", "minimum_speech_seconds"):
            _non_negative(getattr(s, name), f"silence.{name}")
        if s.long_pause_seconds < s.short_pause_seconds:
            raise ConfigError("silence.long_pause_seconds must be >= silence.short_pause_seconds")
        if s.threshold_db > 0:
            raise ConfigError("silence.threshold_db must be <= 0")
        _choice(y.method, ("audio_cross_correlation",), "synchronization.method")
        _positive(y.max_offset_seconds, "synchronization.max_offset_seconds")
        _positive(y.envelope_rate_hz, "synchronization.envelope_rate_hz")
        if not 0 <= y.min_confidence <= 1:
            raise ConfigError("synchronization.min_confidence must be in [0, 1]")
        _choice(sp.diarizer, ("none", "segments_file", "pyannote"), "speakers.diarizer")
        if sp.diarizer == "segments_file" and not sp.segments_file:
            raise ConfigError("speakers.segments_file is required when speakers.diarizer is 'segments_file'")
        for label, player in sp.mapping.items():
            _choice(player, PLAYERS, f"speakers.mapping[{label!r}]")


def _positive(v: float, name: str) -> None:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise ConfigError(f"{name} must be a positive number, got {v!r}")


def _non_negative(v: float, name: str) -> None:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise ConfigError(f"{name} must be a non-negative number, got {v!r}")


def _choice(v: Any, options: tuple[str, ...], name: str) -> None:
    if v not in options:
        raise ConfigError(f"{name} must be one of {list(options)}, got {v!r}")


def _build(cls, data: dict[str, Any], section: str):
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{section}' must be a JSON object")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build and validate a Config from a (possibly partial) dictionary."""
    sections = {"output": OutputConfig, "editing": EditingConfig, "silence": SilenceConfig,
                "synchronization": SynchronizationConfig, "speakers": SpeakersConfig}
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {sorted(unknown)}")
    cfg = Config(**{name: _build(cls, data[name], name) for name, cls in sections.items() if name in data})
    cfg.validate()
    return cfg


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {p}: {exc}") from None


def load_config(path: str | Path | None) -> Config:
    """Load a config file; with ``None`` return the defaults."""
    if path is None:
        cfg = Config()
        cfg.validate()
        return cfg
    return config_from_dict(load_json(path))
