"""Deterministic rules-based camera selection producing an :class:`EDL`.

Rules (all thresholds come from :class:`~src.config.EditingConfig`):

* Exactly one mapped player speaking -> prefer that player's POV, but only
  after the speaker has been the sole active speaker for
  ``speaker_switch_threshold_seconds`` (brief interjections are ignored).
  The switch is placed at the start of that speech.
* Overlapping speech -> keep the current POV.
* Nobody speaking -> keep the current POV, unless gameplay activity is supplied,
  enabled, and the other player is more active by ``activity_switch_margin``.
* Shots last at least ``minimum_shot_seconds`` and speaker switches respect
  ``speaker_switch_cooldown_seconds``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

from .config import EditingConfig, SilenceConfig
from .edit_decision_list import EDL, Segment
from .timeline import PLAYERS, SyncMapping

log = logging.getLogger(__name__)

# (player, start, end) -> mean activity score in [0, 1], or None when unknown
ActivityFn = Callable[[str, float, float], "float | None"]


@dataclass
class SpeechSpan:
    """A speaker interval on the shared timeline. ``player`` is None if the label is unmapped."""
    start: float
    end: float
    label: str
    player: str | None
    confidence: float = 1.0


def _other(p: str) -> str:
    return PLAYERS[1] if p == PLAYERS[0] else PLAYERS[0]


def _make_edl(mapping: SyncMapping, sources: dict[str, str], switches: list[tuple[float, str, str]],
              r0: float, r1: float, spans: list[SpeechSpan], activity: ActivityFn | None,
              audio_player: str, metadata: dict) -> EDL:
    segments: list[Segment] = []
    bounds = [s[0] for s in switches] + [r1]
    for i, (t0, player, reason) in enumerate(switches):
        t1 = bounds[i + 1]
        if t1 - t0 <= 1e-6:
            continue
        overlapping = [s for s in spans if s.end > t0 and s.start < t1]
        labels = sorted({s.label for s in overlapping})
        confs = [s.confidence for s in overlapping]
        act = activity(player, t0, t1) if activity else None
        src0, dur = mapping.shared_to_source(player, t0), t1 - t0
        asrc0 = mapping.shared_to_source(audio_player, t0)
        segments.append(Segment(
            output_start=round(t0 - r0, 6), output_end=round(t1 - r0, 6), player=player,
            source_file=sources[player], source_start=round(src0, 6), source_end=round(src0 + dur, 6),
            shared_start=round(t0, 6), shared_end=round(t1, 6), speakers=labels, activity=act, reason=reason,
            confidence=min(confs) if confs else None, action="retained",
            audio={"source_player": audio_player, "source_file": sources[audio_player],
                   "source_start": round(asrc0, 6), "source_end": round(asrc0 + dur, 6), "mode": "continuous"},
        ))
    # make output contiguous after rounding
    for a, b in zip(segments, segments[1:]):
        b.output_start = a.output_end
    src = {p: {"path": sources[p], "duration": mapping.duration(p), "offset": mapping.offset(p)} for p in PLAYERS}
    return EDL(sources=src, segments=segments, metadata=metadata)


def find_cuts(spans: list[SpeechSpan], r0: float, r1: float, scfg: SilenceConfig) -> list[tuple[float, float]]:
    """Conservative dead-time cuts on the shared timeline.

    Only pauses *between* two speech spans longer than ``long_pause_seconds`` (and at least
    ``minimum_removal_seconds``) are removed. ``short_pause_seconds`` of the pause is kept after the previous speech
    and ``lead_in_seconds`` before the next speech, so the edit cuts in shortly before speaking resumes. Leading and
    trailing silence, shorter pauses, and everything when no speech is known are left alone.
    """
    tail, lead = scfg.short_pause_seconds, scfg.lead_in_seconds
    threshold = max(scfg.minimum_removal_seconds, scfg.long_pause_seconds, tail + lead + 1e-6)
    ivs = sorted((max(s.start, r0), min(s.end, r1)) for s in spans if s.end > r0 and s.start < r1)
    cuts: list[tuple[float, float]] = []
    reach = None
    for a, b in ivs:
        if reach is not None and a - reach > threshold:
            cuts.append((reach + tail, a - lead))
        reach = b if reach is None else max(reach, b)
    return cuts


def apply_cuts(edl: EDL, cuts: list[tuple[float, float]], r0: float) -> EDL:
    """Remove shared-time intervals from an EDL, re-timing the output and recording what was removed."""
    if not cuts:
        return edl
    out: list[Segment] = []
    removed: list[dict] = []
    t = 0.0
    for seg in edl.segments:
        pieces = [(seg.shared_start, seg.shared_end)]
        cut_here = False
        for c0, c1 in cuts:
            nxt = []
            for a, b in pieces:
                if c1 <= a or c0 >= b:
                    nxt.append((a, b))
                    continue
                cut_here = True
                if c0 > a:
                    nxt.append((a, c0))
                if c1 < b:
                    nxt.append((c1, b))
            pieces = nxt
        for a, b in pieces:
            if b - a <= 1e-6:
                continue
            d = b - a
            new = Segment(**{**seg.__dict__, "audio": dict(seg.audio)})
            new.shared_start, new.shared_end = round(a, 6), round(b, 6)
            new.source_start = round(seg.source_start + (a - seg.shared_start), 6)
            new.source_end = round(new.source_start + d, 6)
            if new.audio:
                new.audio["source_start"] = round(seg.audio["source_start"] + (a - seg.shared_start), 6)
                new.audio["source_end"] = round(new.audio["source_start"] + d, 6)
            new.output_start = round(t, 6)
            t += d
            new.output_end = round(t, 6)
            if cut_here:
                new.action = "shortened"
            out.append(new)
    for c0, c1 in cuts:
        removed.append({"shared_start": round(c0, 6), "shared_end": round(c1, 6), "duration": round(c1 - c0, 6),
                        "reason": "long pause between speech removed"})
    for a, b in zip(out, out[1:]):
        b.output_start = a.output_end
    edl.segments, edl.removed = out, removed
    edl.metadata["removed_seconds"] = round(sum(c1 - c0 for c0, c1 in cuts), 6)
    return edl


def plan_unedited(mapping: SyncMapping, sources: dict[str, str], player: str = "player_a",
                  audio_player: str = "player_a") -> EDL:
    """Synchronized, uncut render of a single POV (debugging aid)."""
    r0, r1 = mapping.common_range()
    return _make_edl(mapping, sources, [(r0, player, "unedited synchronized render")], r0, r1, [], None,
                     audio_player, {"mode": "unedited"})


def plan_edit(mapping: SyncMapping, sources: dict[str, str], spans: list[SpeechSpan], cfg: EditingConfig,
              audio_player: str = "player_a", activity: ActivityFn | None = None,
              silence: SilenceConfig | None = None) -> EDL:
    """Create an EDL choosing a POV over the common shared range."""
    r0, r1 = mapping.common_range()
    dt = cfg.planning_step_seconds
    usable = [s for s in spans if s.player in PLAYERS and s.confidence >= cfg.speaker_confidence_threshold]
    use_activity = bool(activity) and cfg.use_gameplay_activity
    min_gap = max(cfg.minimum_shot_seconds, cfg.speaker_switch_cooldown_seconds)

    current = cfg.default_player
    switches: list[tuple[float, str, str]] = [(r0, current, "initial camera (editing.default_player)")]
    last_switch = r0
    cand: str | None = None
    cand_start = r0
    cand_kind = ""
    first_speech: float | None = None
    n = int(math.ceil((r1 - r0) / dt))
    for k in range(n):
        c0, c1 = r0 + k * dt, min(r0 + (k + 1) * dt, r1)
        mid = (c0 + c1) / 2
        active = {s.player for s in usable if s.start <= mid < s.end}
        if active and first_speech is None:
            first_speech = c0
        want, kind = None, ""
        if len(active) == 1:
            want, kind = next(iter(active)), "speaker"
        elif not active and use_activity:
            cur_a = activity(current, c0, c1)  # type: ignore[misc]
            oth_a = activity(_other(current), c0, c1)  # type: ignore[misc]
            if cur_a is not None and oth_a is not None and oth_a - cur_a >= cfg.activity_switch_margin:
                want, kind = _other(current), "activity"
        if want is None or want == current:
            cand = None
            continue
        if cand != want:
            cand, cand_start, cand_kind = want, c0, kind
        if c1 - cand_start < cfg.speaker_switch_threshold_seconds - 1e-9:
            continue
        if len(switches) == 1 and kind == "speaker" and first_speech is not None and cand_start <= first_speech + 1e-9:
            switches[0] = (r0, want, f"initial camera follows first speaker ({want})")
            current, cand = want, None
        elif c1 - last_switch >= min_gap - 1e-9:
            t = max(cand_start, last_switch + min_gap)
            if t >= r1 - cfg.minimum_shot_seconds + 1e-9:
                break
            if cand_kind == "speaker":
                reason = f"speaker change: {want} sole speaker for >= {cfg.speaker_switch_threshold_seconds:g}s"
            else:
                reason = f"no speech; {want} more active (margin >= {cfg.activity_switch_margin:g})"
            switches.append((t, want, reason))
            current, last_switch, cand = want, t, None
    log.info("Planned %d shots over %.1fs", len(switches), r1 - r0)
    edl = _make_edl(mapping, sources, switches, r0, r1, usable, activity, audio_player,
                    {"mode": "auto", "editing": cfg.__dict__.copy()})
    if silence is not None and cfg.remove_silence:
        cuts = find_cuts(spans, r0, r1, silence)
        if cuts:
            log.info("Shortening %d long pause(s), removing %.1fs", len(cuts), sum(b - a for a, b in cuts))
        edl = apply_cuts(edl, cuts, r0)
    return edl
