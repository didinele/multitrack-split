import librosa
import numpy as np
import time

def extract_features(audio_path: str, sr: int = 2000, hop_length: int = 200):
    """
    Loads the analysis audio file and extracts time-series features (RMS, spectral flux, onset strength).
    Every 100ms given sr=2000 and hop_length=200.
    """
    print(f"Loading analysis audio for feature extraction: {audio_path}")
    print(f"  analysis sample rate={sr}, hop_length={hop_length}")

    start_time = time.monotonic()

    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    print(f"  loaded {len(y)} samples ({len(y) / sr:.2f}s at {sr} Hz)")
    
    # 1. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    print(f"  computed RMS energy ({len(rms)} frames)")
    
    # 2. Spectral Flux (Onset strength envelope)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    print(f"  computed onset strength ({len(onset_env)} frames)")
    
    # Normalize features to 0-1 range for easier thresholding
    rms_norm = rms / (np.max(rms) + 1e-10)
    onset_norm = onset_env / (np.max(onset_env) + 1e-10)
    
    # Combine features (e.g., equally weighted)
    combined = 0.5 * rms_norm + 0.5 * onset_norm
    
    # Generate time array for the frames
    times = librosa.frames_to_time(np.arange(len(combined)), sr=sr, hop_length=hop_length)

    print(f"  normalized RMS range: {rms_norm.min():.4f}..{rms_norm.max():.4f}")
    print(f"  normalized onset range: {onset_norm.min():.4f}..{onset_norm.max():.4f}")
    print(f"  combined feature range: {combined.min():.4f}..{combined.max():.4f}")
    print(f"  produced {len(times)} frame timestamps")
    print(f"  feature extraction completed in {time.monotonic() - start_time:.2f}s")
    
    return times, combined, rms_norm, onset_norm
