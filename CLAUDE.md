# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python -m src.cli split
```

There are no tests and no linter configured. Dependencies are in `requirements.txt`; install into the project's `.venv`:

```bash
pip install -r requirements.txt
```

ffmpeg must be on `PATH` (used by `ffmpeg-python` at runtime).

## Architecture

The tool opens a PySide6 GUI (`src/gui/`) that drives a two-phase pipeline:

**Phase 1 — heavy analysis (background thread):**
`AnalysisWorker` (`worker.py`) checks for a cached result, then if needed: builds a 2 kHz mono downmix of all input stems via ffmpeg (`downmix.py`), feeds it to librosa for RMS + onset-strength extraction (`features.py`), and saves the result as a compressed `.npz` (`cache.py`). The worker emits Qt signals to communicate progress and completion back to the main thread.

**Phase 2 — segmentation (main thread, instant):**
`_on_rerun_segmentation` in `window.py` reads the cached numpy arrays and runs the heuristics synchronously: `smooth_features` → `apply_hysteresis` → `get_regions` → `extend_regions_for_ascending_start` → `add_padding` (all in `segmentation.py`). Because only numpy arrays are needed (no I/O), re-running after a parameter change is fast enough to do on the main thread without blocking.

**Canvas (`canvas.py`):**
`WaveformCanvas` is a plain `QWidget` (no matplotlib). It renders the waveform by downsampling the `combined` feature array into pixel-width buckets via `np.maximum.reduceat`, then builds a `QImage` from a numpy palette LUT in `paintEvent`. The `_active_cache` (which rows are "on" per column) is recomputed only on resize or new data — drags only recompute the region mask, keeping repaints fast. Drag handles are always active; hit-testing and clamping are done entirely in Qt mouse events.

**Sidebar (`sidebar.py`):**
All parameters (dirs, thresholds, padding, exclusions, no-cache) live here. Settings are persisted via `QSettings("wav-split", "wav-split")` on close, including `input_dir`.

**Export (`export.py`):**
`export_song` trims every input stem to the confirmed region using `ffmpeg -ss / -t / -c copy` (no re-encoding), running stems in parallel via `ThreadPoolExecutor`. Futures must be awaited with `.result()` to surface exceptions.

## Key constants

- Analysis sample rate: **SR = 2000 Hz**, hop length: **HOP_LENGTH = 200** → one feature frame every 100 ms (defined in `worker.py`, imported by `window.py`).
- Ascending-start heuristic thresholds (`MIN_RISE`, `MAX_DOWN_FRAC`, `MAX_NEGATIVE_STEP`) live in `segmentation.py` — intentionally not exposed in the GUI.

## Cache

Lives in `<input_dir>/.wav_split_cache/`. Key = SHA-256 of all input file contents + names + sizes + mtimes. Cache check (hashing) runs inside the worker thread to avoid blocking the GUI. Invalidate with the "Ignore cache" checkbox.

## Windows-specific notes

- `downmix.py` uses `glob("*.[wW][aA][vV]")` (bracket pattern) to avoid double-listing files on case-insensitive filesystems.
- ffmpeg must be on `PATH`; on Windows this requires manual installation and PATH configuration.
- `ThreadPoolExecutor` futures must always be `.result()`-ed — silently swallowed exceptions were a prior bug.
