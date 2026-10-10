# Minecraft Two-Player POV Editor

Command-line tool that combines two Minecraft POV recordings of the same voice chat into one MP4: it synchronizes
them from their shared audio, detects speech, chooses which POV to show with deterministic rules, writes an
inspectable edit decision list (EDL, JSON) and renders with FFmpeg. Everything runs locally.

## Status

| Feature | State |
|---|---|
| Media inspection (FFprobe), validation, error messages | implemented |
| Audio extraction, automatic sync (envelope cross-correlation), confidence, manual offset, drift check | implemented |
| Energy-based voice activity detection with padding | implemented |
| Speaker diarization via adapters: `segments_file` (manual/precomputed), `pyannote` (optional), `none` | implemented (pyannote untested here – needs model access) |
| Transcription (faster-whisper, optional, `--transcribe`) | implemented, untested here (needs model download) |
| Rules-based POV switching, EDL generation/validation, reasons per segment | implemented |
| FFmpeg render (H.264/AAC, one continuous audio track, atomic output) | implemented |
| Caching keyed on file metadata + config | implemented |
| Gameplay activity (OpenCV) | **not implemented**; the planner accepts an activity function (`ActivityFn`) as extension point |
| Conservative pause removal | implemented: pauses between speech longer than `silence.long_pause_seconds` (default 25 s, and at least `silence.minimum_removal_seconds`) are removed, keeping `silence.short_pause_seconds` after the previous speech and `silence.lead_in_seconds` (3 s) before the next, so the cut-in is not jarring (disable with `editing.remove_silence`); leading/trailing silence and shorter pauses are kept |
| Piecewise drift correction, GUI, semantic events | roadmap |

## Architecture

`src/` modules: `config` (validated JSON), `media_info` (ffprobe), `audio_extraction`, `synchronize`, `timeline`
(offset convention/mapping), `audio_analysis` (VAD, transcription), `speaker_detection` (diarizer adapters),
`edit_planner` (camera rules), `edit_decision_list` (schema + validation), `renderer`, `pipeline` (stages used by
the CLI and any future GUI), `cli`, `main`.

## Windows installation

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH") and Git.
2. Install FFmpeg (includes FFprobe): `winget install Gyan.FFmpeg`, or download a build from https://ffmpeg.org,
   unzip it and add its `bin` folder to `PATH`. Check with `ffmpeg -version` and `ffprobe -version` in a new terminal.
   Alternatively set `FFMPEG_PATH` / `FFPROBE_PATH` to the executables.
3. In the repository folder:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Optional models

* **Transcription**: `pip install faster-whisper`. The Whisper model is downloaded once on first use (internet needed);
  a GPU (CUDA) is optional and speeds it up.
* **Diarization**: `pip install pyannote.audio`, create a Hugging Face account, accept the terms of
  `pyannote/speaker-diarization-3.1`, create an access token and set it in your shell (`set HF_TOKEN=...`). Never put the
  token in files. A GPU is optional. If unavailable, the tool logs a warning and continues with speech detection only.
* Without a diarizer, use `--speaker-segments file.json` (a list of `{"start","end","label","confidence"}` in the audio
  source's time) or accept that the camera stays on `editing.default_player`.

## Input layout

```
input/player_a.mp4   input/player_b.mp4   project/settings.json (copy of project/settings.example.json)
```
`input/`, `output/`, `cache/` contents are git-ignored.

## Usage

```
python -m src.main inspect --player-a input/player_a.mp4 --player-b input/player_b.mp4
python -m src.main synchronize --player-a input/player_a.mp4 --player-b input/player_b.mp4 [--offset -2.5] [--check-drift]
python -m src.main analyze --project project/settings.json
python -m src.main plan    --project project/settings.json --map SPEAKER_00=player_a --map SPEAKER_01=player_b
python -m src.main render  --project project/settings.json            # or --edl cache/edl.json (after hand-editing)
python -m src.main run --player-a input/player_a.mp4 --player-b input/player_b.mp4 --output output/final.mp4
python -m src.main validate cache/edl.json
```
Useful flags: `--offset`, `--map`, `--diarizer`, `--speaker-segments`, `--no-silence-removal`, `--no-activity`,
`--unedited` (synchronized, uncut Player A render for debugging), `--config`, `--work-dir`, `-v`.
`synchronize` writes `<work-dir>/sync_report.json`; `plan` writes `<work-dir>/edl.json`. Diarization labels
(`SPEAKER_00`…) are **never** assumed to be Player A/B; map them with `--map` or `speakers.mapping`.
Exit code is 0 on success, 1 on a handled error.

## Offset and timestamp conventions

* Shared timeline: seconds, zero = start of Player A's recording.
* `source_time = shared_time + offset`. Offset A = 0. If B started recording 5 s after A, B's t=0 is shared t=5, so
  `offset_b = -5`. `--offset` / `synchronization.manual_offset_seconds` use this sign.
* The edit covers the *intersection* of both recordings; output time 0 = start of that intersection.
* EDL fields: `output_*` (final video), `shared_*`, `source_*` (file time). The audio comes from one recording
  (`output.audio_source`) so the voice chat is never doubled.
* Low confidence (< `synchronization.min_confidence`) produces a warning; verify and override with `--offset`.

## Configuration reference (JSON; unknown keys are rejected)

* `output`: `resolution` ("1920x1080", even values), `fps`, `video_codec`, `audio_codec`, `crf`, `preset`, `audio_bitrate`, `audio_source`.
* `editing`: `minimum_shot_seconds` (2.0), `speaker_switch_threshold_seconds` (1.0), `speaker_switch_cooldown_seconds` (3.0),
  `speaker_confidence_threshold` (0.5), `default_player`, `use_gameplay_activity`, `activity_switch_margin`,
  `planning_step_seconds`; reserved: `remove_silence`, `preserve_uncertain_gameplay`.
* `silence`: `threshold_db` (-40), `minimum_speech_seconds`, `speech_padding_seconds`; reserved: `short_pause_seconds`, `long_pause_seconds`, `minimum_removal_seconds`.
* `synchronization`: `method`, `manual_offset_seconds`, `max_offset_seconds`, `min_confidence`, `envelope_rate_hz`.
* `speakers`: `diarizer` (`none|segments_file|pyannote`), `segments_file`, `mapping`.
* Project file additionally has a `project` section: `player_a`, `player_b`, `output`, `work_dir`.

## Camera rules

A player becomes the camera after being the sole speaker for the switch threshold (the cut is placed at the start of
their speech); brief interjections and overlapping speech keep the current POV; shots last at least the minimum shot
length and switches respect the cooldown; the first camera follows the first speaker. Each EDL segment records a reason.

## Known limitations

Constant offset only (drift is reported, not corrected); sync needs shared audio with a distinctive envelope; no
dead-time removal or visual analysis yet; rendering seeks per segment so very many tiny segments are slow; Windows
is the target but the code is cross-platform.

## Troubleshooting

* "'ffmpeg' was not found": fix `PATH` or set `FFMPEG_PATH`/`FFPROBE_PATH`.
* Low sync confidence: pick recordings with clear shared voice chat, or set `--offset` manually.
* Camera never switches: no diarizer/mapping configured (see warning in the log).

## Tests

`pip install -r requirements.txt` then `python -m pytest`. Media tests generate tiny clips and are skipped without FFmpeg.

## Roadmap

OpenCV activity scoring; conservative pause shortening/removal; piecewise drift correction; better speaker-change
rules; semantic gameplay events; review GUI; GPU encoding.
