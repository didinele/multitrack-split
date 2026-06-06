# Multitrack Splitter

A GUI tool to detect and segment songs from long multitrack live recordings.

## Usage

```bash
python -m src.cli
```

This opens an interactive window where you pick your input/output folders, tune parameters, and review the detected song regions before exporting.

## Workflow

1. **Select folders** — pick the input directory (containing multitrack WAV stems) and the output directory for segmented songs.
2. **Run Analysis** — the tool builds a low-rate analysis downmix, extracts energy/onset features, and caches the result for fast iteration. This is the slow step (~seconds to minutes depending on recording length and disk speed).
3. **Review regions** — the waveform canvas shows the audio with detected song regions highlighted. Drag the red boundary handles to adjust any region's start or end time.
4. **Re-run Segmentation** — tweak any parameter and click this to instantly re-apply the heuristics without re-analysing the audio.
5. **Confirm & Export** — exports each song as a numbered folder (`01/`, `02/`, …) containing one trimmed WAV per input stem.

Closing the window or clicking Cancel at any point exits without writing any files.

## Parameters

All parameters live in the GUI sidebar and are persisted between sessions.

### Exclusions
Default: `click`

Comma-separated keywords. Any input file whose name contains one of these (case-insensitive) is excluded from the analysis mix. Useful for tracks like click or drone that carry sustained energy unrelated to musical activity. Excluded stems are still exported for each song.

### Start threshold
Default: `0.3` (range 0.0–1.0)

The normalised signal level the smoothed feature curve must rise above to open a song region (upper hysteresis threshold).

### Stop threshold
Default: `0.1` (range 0.0–1.0)

The level the smoothed curve must fall below to close a song region (lower hysteresis threshold). Should be less than or equal to the start threshold.

### Smooth window
Default: `10 s`

Moving-average window applied to the combined RMS + onset feature before thresholding. Larger values produce a more stable curve that ignores short spikes; smaller values are more responsive but may cause false splits in dynamic recordings.

### Min active
Default: `60 s`

Minimum duration for a detected region to be kept. Raise this to suppress false positives from brief noise bursts like tuning or sound-check activity.

### Min silence
Default: `30 s`

Minimum gap between two songs. Raise this if the tool merges songs that are close together; lower it if it misses splits between back-to-back songs.

### Pre-pad / Post-pad
Default: `25 s` each

Seconds added unconditionally before and after each detected boundary. A safety margin to ensure song intros and tails are not clipped by the energy-based boundary.

## Caching

After the first analysis run, the extracted features are cached keyed by a SHA-256 hash of the input file metadata (name, size, mtime). Subsequent runs with the same inputs skip the downmix and feature extraction entirely and load from cache, making parameter iteration fast regardless of recording length.

On an M3 Pro MacBook with ~19 GB of audio (10 stems):

| Scenario | Time |
|---|---|
| Cold run | ~20 s |
| Cache hit | < 0.1 s |

## Output

Each confirmed song is exported to a numbered subdirectory of the output folder:

```
output/
  01/
    bass.wav
    drums.wav
    guitar.wav
    ...
  02/
    ...
```

All stems are trimmed to exactly the confirmed region boundaries.

## Backlog

- [x] Downmix preview track for better altering the detected regions.
- [ ] Look into bundling the app fully to remove the need for a Python runtime installed.
- [x] Replaced full content-hash cache with a metadata cache (name + size + mtime). A false positive requires
deliberately crafting identical metadata with different audio, so the heavier hash buys nothing in practice.
- [x] Add support for zooming into the region chart.
- [ ] App icon
- [ ] Audio preview will only use the audio devices that was selected when the app was open
