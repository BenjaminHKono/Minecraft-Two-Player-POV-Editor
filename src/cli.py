"""Command-line interface. All commands call :mod:`src.pipeline`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import pipeline
from .edit_decision_list import EDL
from .errors import ConfigError, EditorError
from .logging_config import setup_logging
from .synchronize import SyncResult

log = logging.getLogger(__name__)


def _common(p: argparse.ArgumentParser, inputs: bool = True) -> None:
    p.add_argument("--project", help="project settings JSON (inputs + configuration)")
    p.add_argument("--config", help="configuration JSON (defaults used if omitted)")
    if inputs:
        p.add_argument("--player-a", help="Player A recording (defines the shared timeline)")
        p.add_argument("--player-b", help="Player B recording")
    p.add_argument("--work-dir", help="cache/working directory (default: cache)")
    p.add_argument("--offset", type=float, help="manual offset of Player B in seconds: source_B = shared + offset")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def _edit_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--map", action="append", default=[], metavar="LABEL=PLAYER",
                   help="map a diarization label to player_a/player_b (repeatable), e.g. SPEAKER_00=player_a")
    p.add_argument("--diarizer", choices=["none", "segments_file", "pyannote"], help="speaker diarization backend")
    p.add_argument("--speaker-segments", help="JSON file with speaker segments (implies --diarizer segments_file)")
    p.add_argument("--no-silence-removal", action="store_true", help="disable silence/dead-time removal")
    p.add_argument("--no-activity", action="store_true", help="disable gameplay-activity-based camera selection")
    p.add_argument("--unedited", action="store_true", help="synchronized, uncut render of Player A (debugging)")
    p.add_argument("--transcribe", action="store_true", help="also transcribe with faster-whisper (optional)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.main", description="Automated Minecraft two-player POV editor")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="show media properties of both recordings"); _common(p)
    p = sub.add_parser("synchronize", help="estimate the offset between the recordings"); _common(p)
    p.add_argument("--check-drift", action="store_true", help="also check for drift between recordings")
    p = sub.add_parser("analyze", help="detect speech and speakers in the shared audio"); _common(p); _edit_opts(p)
    p = sub.add_parser("plan", help="create the edit decision list (JSON)"); _common(p); _edit_opts(p)
    p.add_argument("--edl", help="where to write the EDL (default: <work-dir>/edl.json)")
    p = sub.add_parser("render", help="render the MP4 from an EDL (or plan first if none exists)")
    _common(p); _edit_opts(p)
    p.add_argument("--edl", help="existing EDL to render (skips planning)")
    p.add_argument("--output", help="output MP4 path")
    p = sub.add_parser("run", help="run the whole pipeline"); _common(p); _edit_opts(p)
    p.add_argument("--output", help="output MP4 path")
    p = sub.add_parser("validate", help="validate an EDL file"); p.add_argument("edl")
    p.add_argument("-v", "--verbose", action="store_true")
    return ap


def _project(args: argparse.Namespace) -> pipeline.Project:
    overrides = {"player_a": getattr(args, "player_a", None), "player_b": getattr(args, "player_b", None),
                 "work_dir": getattr(args, "work_dir", None), "output": getattr(args, "output", None)}
    project = pipeline.load_project(args.project, overrides, args.config)
    cfg = project.config
    for item in getattr(args, "map", []):
        label, sep, player = item.partition("=")
        if not sep:
            raise ConfigError(f"--map expects LABEL=PLAYER, got {item!r}")
        cfg.speakers.mapping[label] = player
    if getattr(args, "speaker_segments", None):
        cfg.speakers.diarizer, cfg.speakers.segments_file = "segments_file", args.speaker_segments
    if getattr(args, "diarizer", None):
        cfg.speakers.diarizer = args.diarizer
    if getattr(args, "no_silence_removal", False):
        cfg.editing.remove_silence = False
    if getattr(args, "no_activity", False):
        cfg.editing.use_gameplay_activity = False
    cfg.validate()
    return project


def _progress(frac: float) -> None:
    print(f"\rRendering: {frac * 100:5.1f}%", end="", file=sys.stderr, flush=True)
    if frac >= 1.0:
        print(file=sys.stderr)


def _print_sync(sync: SyncResult) -> None:
    print(f"Offset of Player B: {sync.offset_b_seconds:+.3f} s  (source_B = shared + offset; shared = Player A time)")
    print(f"Confidence: {sync.confidence:.2f}  method: {sync.method}")
    for w in sync.warnings:
        print(f"WARNING: {w}")
    if sync.diagnostics:
        print("Diagnostics:", json.dumps(sync.diagnostics))


def _print_edl(edl: EDL) -> None:
    print(f"EDL: {len(edl.segments)} segments, {edl.output_duration:.2f}s")
    for i, s in enumerate(edl.segments):
        print(f"  {i:3d} out {s.output_start:8.2f}-{s.output_end:8.2f}  {s.player}  src {s.source_start:8.2f}-{s.source_end:8.2f}  [{s.action}] {s.reason}")


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "validate":
        edl = EDL.load(args.edl)
        print(f"EDL OK: {len(edl.segments)} segments, {edl.output_duration:.2f}s")
        return 0
    project = _project(args)
    manual = args.offset
    if args.command == "inspect":
        for p, info in pipeline.inspect(project).items():
            print(f"{p}: {json.dumps(info.to_dict(), indent=2)}")
        return 0
    if args.command == "synchronize":
        pipeline.inspect(project)
        sync = pipeline.synchronize(project, manual, args.check_drift)
        _print_sync(sync)
        (project.work / "sync_report.json").write_text(json.dumps(sync.to_dict(), indent=2), encoding="utf-8")
        return 0
    if args.command == "run":
        out, edl, sync = pipeline.run_all(project, manual, args.unedited, args.transcribe, _progress)
        _print_sync(sync)
        _print_edl(edl)
        print(f"Output: {out}")
        return 0
    pipeline.inspect(project)
    sync = pipeline.synchronize(project, manual)
    _print_sync(sync)
    if args.command == "analyze":
        a = pipeline.analyze(project, sync, args.transcribe)
        labels = sorted({s["label"] for s in a["speakers"]})
        print(f"Speech intervals: {len(a['speech'])}; speaker labels: {labels}")
        print("Map labels to players with --map LABEL=player_a|player_b (labels are NOT assumed to match players).")
        return 0
    analysis = None if args.unedited else pipeline.analyze(project, sync, args.transcribe)
    if args.command == "plan":
        edl = pipeline.plan(project, sync, analysis, args.unedited, args.edl or project.work / "edl.json")
        _print_edl(edl)
        return 0
    if args.command == "render":
        edl = EDL.load(args.edl) if args.edl else pipeline.plan(project, sync, analysis, args.unedited, project.work / "edl.json")
        out = pipeline.render(project, edl, args.output, _progress)
        print(f"Output: {out}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        return dispatch(args)
    except EditorError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted; partial output files were removed")
        return 130
