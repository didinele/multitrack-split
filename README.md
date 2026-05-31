# Multitrack Splitter

A CLI tool to detect and segment songs from long multitrack live recordings.

## Usage

Run the split command with the input folder containing WAV tracks and the output folder for segmented songs.

```bash
python -m src.cli split /path/to/multitrack-session /path/to/output-folder
```

This command will:

- discover the usable WAV files in the input folder
- build a low-rate analysis downmix
- compute smoothed feature curves for energy-based segmentation
- apply dual-threshold hysteresis to find sustained song regions
- extend song starts for ascending intro energy
- export each song into `01`, `02`, etc.
- save `debug.png` in the output folder for visual verification

## CLI Options

Note that all the following defaults were chosen based on common sense and testing with live recordings used by the
author. Your audio files may require vastly different settings. It is highly recommended to inspect the `debug.png` output
to better understand how the algorithm is interpreting your audio and to adjust the following parameters accordingly.

### `--exclusions`
Default: `['click']`

Keywords used to exclude stems from the analysis downmix. You may want to use this to remove tracks that might produce
sustained energy, but are not reliable indicators of song presence within your recordings (e.g. a click track running
for minutes on end before the musical moment actually starts, a drone track, or a noisy signal).

Note that the exclusion is rather naive and simply checks if the specified keywords (lowercased) are present
in any of the filenames (lowercased). If a file matches any of the keywords, it is excluded from the downmix used for segmentation, but it will still be copied to the output folder for each song.

### `--start-thresh`
Default: `0.3`

The upper hysteresis (0.0 to 1.0) threshold that triggers the tool to enter an active song region.

0.3 is a sensible default from my testing, but the correct value can vary widely based on recording. If the tool is
not behaving as you expect, analyze the `debug.png` chart and adjust this threshold accordingly.

### `--stop-thresh`
Default: `0.1`

The lower hysteresis (0.0 to 1.0) threshold that triggers exiting an active song region.

Note that you almost certainly do not want this to be higher than `--start-thresh`.

### `--smooth-windows-sec`
Default: `10`

When downmixing the multitrack session to a single analysis signal, the tool applies a moving average smoothing to create
a more stable curve for thresholding. This parameter controls the size of the smoothing window in seconds.

Larger values will create a smoother curve that is less sensitive to short transient spikes. Smaller values will create
a more responsive curve that may be more accurate for tight segues, but it can also create more false splits if the
recording is noisy or has a lot of dynamic variation within songs.

### `--min-active-sec`
Default: `60`

The minimum duration required for a detected region to be considered a song.

Raising this can be helpful in eliminating false positives from short noise bursts like tuning, soundchecks, etc.
You'd naturally want to set this lower if you are working with short songs.

### `--min-silence-sec`
Default: `30`

The minimum silence gap needed to split two songs.

Raising this can be helpful to prevent false splits from brief quiet moments within a song, but it can also cause
the tool to merge two distinct songs into one if they are spaced closely together.

### `--pre-pad`
Default: `25`

Since all tuning can be a double-edged sword (improving certain tracks in your recording, making it worse for others),
the tuning options provide a way to forcefully extend the detected song boundaries in a way that is not influenced by the audio content. This can be useful to ensure that the intro or tail of a song is preserved, even if the energy-based analysis fails to capture it.

### `--post-pad`
Default: `25`

Since all tuning can be a double-edged sword (improving certain tracks in your recording, making it worse for others),
the tuning options provide a way to forcefully extend the detected song boundaries in a way that is not influenced by the audio content. This can be useful to ensure that the intro or tail of a song is preserved, even if the energy-based analysis fails to capture it.

### `--no-output`
Default: `False`

When set, the tool runs the full analysis and still saves the debug chart, but it skips writing segmented WAVs and metadata.

Useful when you want to quickly iterate the analysis parameters and visually verify the results without waiting for the
export process, which is I/O bound and potentially slow.

### `--no-cache`
Default: `False`

When set, the tool ignores cached analysis and recomputes the downmix and audio features from scratch.

As a sort of example of the cache optimizations, on my MacBook M3 Pro, the runtime of the tool on ~19GBs of audio,
(made up of 10 tracks) is around 37 seconds on a cold run with caching enabled.

As the cache is deeply safe (it ensures the input is absolutely identical - it's up for discussion whether this is worth
it, or is behavior that should perhaps be configurable via an option), providing `--no-cache` lowers the runtime down
to around 26 seconds.

This is a significant improvement, but if we now look at a run with a cache hit, the runtime is down to 12 seconds.
I cannot currently confirm if this behavior scales perfectly with the size of the input (and disk read speeds could
make the current caching algorithm terrible on your system), but it is clear that the caching mechanism, even in its
current form, provides a significant speed boost when iterating on the analysis parameters without changing the input
files.

That being said, if you have a particularly large input and are confident you can one-shot the settings, providing `--no-cache` will significantly speed up the runtime, but the results are diminished even within 3 total runs.

## Output and Debugging

The tool always generates `debug.png` in the output folder.

The chart shows:

- the combined energy/onset analysis signal
- the smoothed feature curve used for thresholding
- the binary hysteresis state
- raw candidate region boundaries
- detected ascending intro extensions
- final padded export regions

This visual output is the primary way to verify how the algorithm interpreted the signal and where song boundaries were placed.
