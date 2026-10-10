"""Pipeline stages shared by every CLI command (and a future GUI)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import audio_analysis as aa
from .audio_extraction import extract_audio
from .config import Config, config_from_dict, load_json
from .edit_decision_list import EDL
from .edit_planner import SpeechSpan, plan_edit, plan_unedited
from .errors import ConfigError, EditorError
from .media_info import MediaInfo, probe
from .renderer import render as render_edl
from .speaker_detection import SpeakerSegment, make_diarizer
from .synchronize import SyncResult, check_drift, manual_result, synchronize_audio
from .timeline import PLAYERS, SyncMapping

log = logging.getLogger(__name__)
PROJECT_KEYS = {"player_a", "player_b", "output", "work_dir"}


@dataclass
class Project:
    player_a: str
    player_b: str
    config: Config = field(default_factory=Config)
    output: str = "output/final.mp4"
    work_dir: str = "cache"
    media: dict[str, MediaInfo] = field(default_factory=dict)

    def path(self, player: str) -> str:
        return self.player_a if player == "player_a" else self.player_b

    @property
    def work(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_project(path: str | Path | None, overrides: dict[str, Any] | None = None,
                config_path: str | Path | None = None) -> Project:
    """Build a Project from a project file (JSON with a ``project`` section + config sections)."""
    data: dict[str, Any] = {}
    if config_path:
        data = load_json(config_path)
    proj: dict[str, Any] = {}
    if path:
        file_data = load_json(path)
        proj = file_data.pop("project", {})
        data.update(file_data)
    data.pop("project", None)
    unknown = set(proj) - PROJECT_KEYS
    if unknown:
        raise ConfigError(f"Unknown key(s) in 'project': {sorted(unknown)}")
    for k, v in (overrides or {}).items():
        if v is not None:
            proj[k] = v
    missing = [k for k in ("player_a", "player_b") if not proj.get(k)]
    if missing:
        raise ConfigError(f"Missing input(s): {missing}. Provide --player-a/--player-b or a project file.")
    return Project(player_a=proj["player_a"], player_b=proj["player_b"], config=config_from_dict(data),
                   output=proj.get("output", "output/final.mp4"), work_dir=proj.get("work_dir", "cache"))


def inspect(project: Project) -> dict[str, MediaInfo]:
    for p in PLAYERS:
        project.media[p] = probe(project.path(p))
    return project.media


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sig(info: MediaInfo) -> list:
    return [info.path, info.size_bytes, info.mtime]


def _file_sig(path: str | None) -> list | None:
    if not path or not Path(path).is_file():
        return None
    st = Path(path).stat()
    return [path, st.st_size, st.st_mtime]


def _cached_json(path: Path, key: str, producer: Callable[[], Any]) -> Any:
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                log.info("Using cached %s", path.name)
                return stored["data"]
        except (json.JSONDecodeError, OSError):
            pass
    data = producer()
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({"key": key, "data": data}, indent=2), encoding="utf-8")
    tmp.replace(path)
    return data


def _wav(project: Project, player: str) -> Path:
    info = project.media[player]
    wav = project.work / f"{player}.wav"
    marker = wav.with_suffix(".key")
    key = _key(_sig(info))
    if not (wav.is_file() and marker.is_file() and marker.read_text() == key):
        extract_audio(info, wav)
        marker.write_text(key)
    return wav


def synchronize(project: Project, manual_offset: float | None = None, detect_drift: bool = False) -> SyncResult:
    """Return the sync result; a manual offset (argument or config) bypasses estimation."""
    if not project.media:
        inspect(project)
    cfg = project.config.synchronization
    manual = manual_offset if manual_offset is not None else cfg.manual_offset_seconds
    if manual is not None:
        log.info("Using manual offset %+.3fs", manual)
        return manual_result(float(manual))
    key = _key("sync", [_sig(project.media[p]) for p in PLAYERS], cfg.__dict__)
    def produce() -> dict:
        t = time.monotonic()
        res = synchronize_audio(_wav(project, "player_a"), _wav(project, "player_b"), cfg.envelope_rate_hz,
                                cfg.max_offset_seconds, cfg.min_confidence)
        log.info("Synchronization took %.1fs", time.monotonic() - t)
        return res.to_dict()
    data = _cached_json(project.work / "sync.json", key, produce)
    result = SyncResult(**data)
    if detect_drift:
        result.warnings += check_drift(_wav(project, "player_a"), _wav(project, "player_b"),
                                       result.offset_b_seconds, cfg.envelope_rate_hz)
    for w in result.warnings:
        log.warning(w)
    return result


def make_mapping(project: Project, sync: SyncResult) -> SyncMapping:
    return SyncMapping.from_b_offset(sync.offset_b_seconds, project.media["player_a"].duration,
                                     project.media["player_b"].duration)


def analyze(project: Project, sync: SyncResult, transcribe: bool = False) -> dict[str, Any]:
    """Analyze the shared audio once (from ``output.audio_source``); results are on the shared timeline."""
    if not project.media:
        inspect(project)
    cfg = project.config
    mapping = make_mapping(project, sync)
    src_player = cfg.output.audio_source
    off = mapping.offset(src_player)
    key = _key("analysis", _sig(project.media[src_player]), cfg.silence.__dict__, cfg.speakers.__dict__, off, transcribe,
               _file_sig(cfg.speakers.segments_file))

    def produce() -> dict:
        wav = _wav(project, src_player)
        levels, frame = aa.frame_energies_db(wav)
        speech = aa.detect_speech(levels, frame, cfg.silence.threshold_db, cfg.silence.minimum_speech_seconds,
                                  padding_seconds=cfg.silence.speech_padding_seconds)
        log.info("Detected %d speech intervals", len(speech))
        diarizer = make_diarizer(cfg.speakers.diarizer, cfg.speakers.segments_file)
        try:
            segs = diarizer.diarize(wav, speech)
        except EditorError as exc:
            log.warning("Speaker diarization unavailable (%s); continuing with speech detection only", exc)
            segs = make_diarizer("none").diarize(wav, speech)
        out: dict[str, Any] = {
            "audio_source": src_player,
            "speech": [{"start": s.start - off, "end": s.end - off} for s in speech],
            "speakers": [SpeakerSegment(s.start - off, s.end - off, s.label, s.confidence).to_dict() for s in segs],
        }
        if transcribe:
            try:
                tr = aa.transcribe(wav)
                for seg in tr:
                    seg["start"] -= off; seg["end"] -= off
                    for w in seg["words"]:
                        w["start"] -= off; w["end"] -= off
                out["transcript"] = tr
            except EditorError as exc:
                log.warning("Transcription skipped: %s", exc)
        return out

    return _cached_json(project.work / "analysis.json", key, produce)


def speech_spans(analysis: dict[str, Any], mapping_labels: dict[str, str]) -> list[SpeechSpan]:
    return [SpeechSpan(s["start"], s["end"], s["label"], mapping_labels.get(s["label"]), s.get("confidence", 1.0))
            for s in analysis["speakers"]]


def plan(project: Project, sync: SyncResult, analysis: dict[str, Any] | None, unedited: bool = False,
         save_to: str | Path | None = None) -> EDL:
    cfg = project.config
    mapping = make_mapping(project, sync)
    sources = {p: project.path(p) for p in PLAYERS}
    if unedited or analysis is None:
        edl = plan_unedited(mapping, sources, cfg.editing.default_player, cfg.output.audio_source)
    else:
        spans = speech_spans(analysis, cfg.speakers.mapping)
        if not any(s.player for s in spans):
            log.warning("No speaker is mapped to a player (diarizer=%s); camera stays on %s. See README "
                        "'Speaker mapping'.", cfg.speakers.diarizer, cfg.editing.default_player)
        edl = plan_edit(mapping, sources, spans, cfg.editing, cfg.output.audio_source)
    edl.metadata.update({"sync": sync.to_dict(), "output": cfg.output.__dict__.copy()})
    edl.validate()
    if save_to:
        edl.save(save_to)
        log.info("Saved edit decision list to %s", save_to)
    return edl


def render(project: Project, edl: EDL, output: str | None = None, progress: Callable[[float], None] | None = None) -> Path:
    return render_edl(edl, project.config.output, output or project.output, progress)


def run_all(project: Project, manual_offset: float | None = None, unedited: bool = False, transcribe: bool = False,
            progress: Callable[[float], None] | None = None) -> tuple[Path, EDL, SyncResult]:
    t0 = time.monotonic()
    inspect(project)
    sync = synchronize(project, manual_offset)
    analysis = None if unedited else analyze(project, sync, transcribe)
    edl = plan(project, sync, analysis, unedited, project.work / "edl.json")
    out = render(project, edl, progress=progress)
    log.info("Pipeline finished in %.1fs", time.monotonic() - t0)
    return out, edl, sync
