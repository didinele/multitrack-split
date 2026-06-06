import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

def compute_file_hash(path: Path) -> tuple[str, bytes]:
    file_hasher = hashlib.sha256()
    stat = path.stat()
    file_hasher.update(path.name.encode())
    file_hasher.update(b"\0")
    file_hasher.update(str(stat.st_size).encode())
    file_hasher.update(b"\0")
    file_hasher.update(str(stat.st_mtime_ns).encode())
    file_hasher.update(b"\0")

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            file_hasher.update(chunk)

    return str(path), file_hasher.digest()

def compute_input_hash(file_paths: list[Path]) -> str:
    sorted_files = sorted(file_paths, key=lambda p: str(p))

    print(f"Computing cache hash for {len(sorted_files)} input files...")

    start_time = time.monotonic()
    file_hashes: list[tuple[str, bytes]] = []

    max_workers = min(32, len(sorted_files) or 1)

    with ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(compute_file_hash, path): path for path in sorted_files}
        for future in as_completed(futures):
            file_hashes.append(future.result())

    sha = hashlib.sha256()
    for path_str, digest in sorted(file_hashes, key=lambda item: item[0]):
        sha.update(path_str.encode())
        sha.update(b"\0")
        sha.update(digest)
        sha.update(b"\0")

    cache_hash = sha.hexdigest()
    print(f"  input hash computed in {time.monotonic() - start_time:.2f}s")

    return cache_hash

def get_preview_downmix_path(cache_dir: Path, cache_hash: str, sr: int, channels: int) -> Path:
    return cache_dir / f"{cache_hash}_preview_{sr}_{channels}ch.wav"

def get_cache_paths(input_dir: Path, file_paths: list[Path]) -> tuple[Path, Path, str]:
    cache_dir = input_dir / ".multitrack_split_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using cache directory: {cache_dir}")

    cache_hash = compute_input_hash(file_paths)
    cache_data_path = cache_dir / f"{cache_hash}.npz"
    cache_meta_path = cache_dir / f"{cache_hash}.json"

    print(f"Cache key: {cache_hash}")
    print(f"Cache data path: {cache_data_path}")
    print(f"Cache meta path: {cache_meta_path}")

    return cache_data_path, cache_meta_path, cache_hash

def load_cached_analysis(cache_data_path: Path):
    if not cache_data_path.exists():
        print(f"Cache file does not exist: {cache_data_path}")
        return None

    print(f"Loading cached analysis from {cache_data_path}")
    with np.load(cache_data_path) as cached:
        return cached["times"], cached["combined"], cached["rms_norm"], cached["onset_norm"]

def save_cached_analysis(cache_data_path: Path, cache_meta_path: Path, file_paths: list[Path], exclusions: list[str], sr: int, hop_length: int, times, combined, rms_norm, onset_norm):
    print(f"Saving cached analysis to {cache_data_path} and metadata to {cache_meta_path}")
    start_time = time.monotonic()
    np.savez_compressed(cache_data_path,
                        times=times,
                        combined=combined,
                        rms_norm=rms_norm,
                        onset_norm=onset_norm)
    cache_info = {
        "cache_hash": cache_data_path.stem,
        "exclusions": exclusions,
        "file_count": len(file_paths),
        "sample_rate": sr,
        "hop_length": hop_length,
        "files": [str(p.name) for p in file_paths]
    }
    with open(cache_meta_path, "w") as meta_file:
        json.dump(cache_info, meta_file, indent=2)
    print(f"  cached analysis saved in {time.monotonic() - start_time:.2f}s")
