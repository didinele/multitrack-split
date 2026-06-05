import typer
import tempfile
import glob
from pathlib import Path
from typing import List

from .cache import get_cache_paths, load_cached_analysis, save_cached_analysis
from .downmix import create_analysis_downmix, get_audio_files
from .features import extract_features
from .segmentation import smooth_features, apply_hysteresis, get_regions, extend_regions_for_ascending_start, add_padding
from .export import export_song, save_metadata
from .plot import plot_debug

app = typer.Typer(help="Multitrack Splitter: A CLI tool to detect and segment songs from long multitrack live recordings.")

@app.command()
def split(
    input_dir: Path = typer.Argument(..., help="Directory containing the multitrack WAV files"),
    output_dir: Path = typer.Argument(..., help="Directory to save the segmented song files and metadata"),
    exclusions: List[str] = typer.Option(["click"], help="Keywords in filenames to exclude from analysis mix"),
    start_thresh: float = typer.Option(0.3, help="Hysteresis start threshold (0.0 to 1.0)"),
    stop_thresh: float = typer.Option(0.1, help="Hysteresis stop threshold (0.0 to 1.0)"),
    smooth_windows_sec: int = typer.Option(10, help="Smoothing window size (seconds)"),
    min_active_sec: int = typer.Option(60, help="Minimum duration for a section to be considered a song (seconds)"),
    min_silence_sec: int = typer.Option(30, help="Minimum silence between songs to split them"),
    pre_pad: int = typer.Option(25, help="Padding to add before a detected song (seconds)"),
    post_pad: int = typer.Option(25, help="Padding to add after a detected song (seconds)"),
    no_output: bool = typer.Option(False, "--no-output", "-n", help="Skip exporting songs and metadata; generate only debug output and chart for tuning"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached analysis and recompute downmix/features"),
):
    if not input_dir.is_dir():
        typer.echo(f"Error: {input_dir} is not a valid directory.")
        raise typer.Exit(1)

    if no_cache:
        typer.echo("Note: --no-cache is enabled; cached analysis will be ignored.")

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Step 0/5: Discovering audio files in {input_dir}")
    typer.echo(f"  exclusions={exclusions}")
    input_files = get_audio_files(input_dir, exclusions)
    if not input_files:
        typer.echo(f"Error: No valid WAV files found after applying exclusions: {exclusions}")
        raise typer.Exit(1)

    typer.echo(f"  found {len(input_files)} valid audio files for analysis")
    cache_hash = None
    cache_data_path = None
    cache_meta_path = None

    if not no_cache:
        cache_data_path, cache_meta_path, cache_hash = get_cache_paths(input_dir, input_files)

    sr = 2000
    hop_length = 200

    if (not no_cache) and cache_data_path.exists():
        typer.echo(f"Step 1/5: Cache hit for input hash {cache_hash[:8]}; reusing previous feature extraction.")
        typer.echo("Step 2/5: Loading cached analysis data...")
        times, combined, rms_norm, onset_norm = load_cached_analysis(cache_data_path)
    else:
        if no_cache:
            typer.echo("Step 1/5: Cache disabled by --no-cache; computing downmix and features.")
        else:
            typer.echo(f"Step 1/5: Cache miss for input hash {cache_hash[:8]}; computing downmix and features.")

        typer.echo("  creating temporary analysis downmix WAV file...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            downmix_path = Path(tmp_wav.name)
        try:
            create_analysis_downmix(downmix_path, input_files, sr)

            typer.echo("Step 2/5: Extracting audio features from downmix...")
            times, combined, rms_norm, onset_norm = extract_features(str(downmix_path), sr=sr, hop_length=hop_length)

            if not no_cache:
                typer.echo("  saving extracted features to cache for future runs...")
                save_cached_analysis(cache_data_path,
                                     cache_meta_path,
                                     input_files,
                                     exclusions,
                                     sr,
                                     hop_length,
                                     times,
                                     combined,
                                     rms_norm,
                                     onset_norm)
        finally:
            if downmix_path.exists():
                downmix_path.unlink()
                typer.echo("  removed temporary downmix file")

    typer.echo("Step 3/5: Applying segmentation heuristics and ascent detection...")
    frames_per_sec = sr // hop_length
    smooth_frames = smooth_windows_sec * frames_per_sec

    typer.echo(f"  using sampling rate={sr} Hz and hop_length={hop_length} samples")
    typer.echo(f"  smoothing window={smooth_windows_sec}s ({smooth_frames} analysis frames)")

    smoothed = smooth_features(combined, smooth_frames)
    typer.echo("  features smoothed with moving average to reduce transient spikes")
    state = apply_hysteresis(smoothed, start_thresh, stop_thresh)
    typer.echo("  hysteresis state computed to capture sustained active regions")

    raw_regions = get_regions(state, times, min_active_sec, min_silence_sec)
    typer.echo(f"  raw segment candidates after thresholding and merge: {len(raw_regions)}")
    extended_regions, ascension_spans = extend_regions_for_ascending_start(raw_regions, times, smoothed, max_lookback_sec=min_silence_sec)
    if len(extended_regions) != len(raw_regions):
        typer.echo(f"  segment count changed after ascension extension: {len(raw_regions)} -> {len(extended_regions)}")
    elif ascension_spans:
        typer.echo(f"  applied ascending intro extension to {len(ascension_spans)} region(s)")

    if ascension_spans:
        typer.echo("  detected ascending intro extensions for the following regions:")
        for old_start, new_start in ascension_spans:
            typer.echo(f"    moved start earlier from {old_start:.2f}s to {new_start:.2f}s")

    max_time = times[-1]
    final_regions = add_padding(extended_regions, pre_pad, post_pad, max_time)
    typer.echo("  applied safety padding to detected regions")

    typer.echo("Step 4/5: Generating segmentation visualization...")
    typer.echo(f"  writing debug chart to {output_dir / 'debug.png'}")
    plot_debug(times, combined, smoothed, state, raw_regions, extended_regions, ascension_spans, str(output_dir / "debug.png"))

    if no_output:
        typer.echo("Skipping song export and metadata save because --no-output is set.")
    else:
        typer.echo("Step 5/5: Exporting multitrack segments...")
        all_wavs = [Path(f) for f in glob.glob(str(input_dir / "*.[wW][aA][vV]"))]

        for idx, region in enumerate(final_regions, 1):
            typer.echo(f"  Exporting Song {idx} ({region[0]:.1f}s to {region[1]:.1f}s)...")
            export_song(idx, region, output_dir, all_wavs)

        save_metadata(final_regions, output_dir)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
