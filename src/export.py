import json
import ffmpeg
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def export_segment_for_file(input_wav: Path, target_dir: Path, start_time: float, duration: float):
    """Trims a single WAV file without re-encoding and saves it to target_dir."""
    output_wav = target_dir / input_wav.name

    # -ss: start time, -t: duration, -c copy: avoid re-encoding
    (
        ffmpeg
        .input(str(input_wav), ss=start_time, t=duration)
        .output(str(output_wav), c='copy', loglevel='error')
        .run(overwrite_output=True)
    )

def export_song(song_idx: int, region: tuple[float, float], output_dir: Path, all_input_wavs: list[Path]):
    """Exports all tracks for a detected song into a numbered subdirectory."""
    start_time, end_time = region
    duration = end_time - start_time

    song_dir = output_dir / f"{song_idx:02d}"
    song_dir.mkdir(parents=True, exist_ok=True)

    # Run exports concurrently for this song's tracks
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(export_segment_for_file, wav, song_dir, start_time, duration) for wav in all_input_wavs]

    for future in futures:
        future.result()

def save_metadata(regions: list[tuple[float, float]], output_dir: Path):
    metadata = {"songs": []}
    for i, (start, end) in enumerate(regions, 1):
        metadata["songs"].append({
            "idx": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3)
        })
        
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
