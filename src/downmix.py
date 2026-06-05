import os
import time
from glob import glob
from pathlib import Path
import ffmpeg

def get_audio_files(input_dir: Path, exclusions: list[str]) -> list[Path]:
    all_files = glob(str(input_dir / "*.[wW][aA][vV]"))
    valid_files = []
    
    for f in sorted(all_files):
        filename = os.path.basename(f).lower()
        if any(excl.lower() in filename for excl in exclusions):
            print(f"Excluding {f} due to exclusion keywords.")
            continue

        valid_files.append(Path(f))

    return valid_files

def create_analysis_downmix(output_file: Path, input_files: list[Path], sample_rate: int):
    """
    Combines selected valid WAV files into a single mono analysis file
    at a low sample rate (default 2kHz).
    """
    print(f"Creating analysis downmix from {len(input_files)} files at {sample_rate} Hz")
    for f in input_files:
        print(f"  include: {f}")
    print(f"  output downmix path: {output_file}")
    print("  mixing tracks into mono analysis signal...")

    inputs = [ffmpeg.input(str(f)) for f in input_files]
    
    # amix filter mixes multiple audio streams into one
    mixed = ffmpeg.filter(inputs, 'amix', inputs=len(inputs), normalize=0)
    
    # Output to a low sample rate mono file
    out = ffmpeg.output(mixed, str(output_file), ac=1, ar=sample_rate, loglevel='error')

    start_time = time.monotonic()
    # Overwrite if exists
    ffmpeg.run(out, overwrite_output=True)
    elapsed = time.monotonic() - start_time

    print(f"  downmix completed in {elapsed:.2f}s")
