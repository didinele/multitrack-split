import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..downmix import create_analysis_downmix
from ..features import extract_features
from ..cache import get_cache_paths, get_preview_downmix_path, load_cached_analysis, save_cached_analysis

SR = 2000
HOP_LENGTH = 200
PREVIEW_SR_DEFAULT = 22050

class AnalysisWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # (times, combined, rms_norm, onset_norm, preview_path)
    error = Signal(str)

    def __init__(
        self,
        input_files: list[Path],
        input_dir: Path,
        exclusions: list[str],
        preview_sr: int = PREVIEW_SR_DEFAULT,
        preview_channels: int = 1,
    ):
        super().__init__()
        self._input_files = input_files
        self._input_dir = input_dir
        self._exclusions = exclusions
        self._preview_sr = preview_sr
        self._preview_channels = preview_channels

    @Slot()
    def run(self):
        analysis_downmix_path = None
        try:
            # Always compute the cache hash so the preview path is always known.
            self.progress.emit("Computing cache hash...")
            cache_data_path, cache_meta_path, cache_hash = get_cache_paths(
                self._input_dir, self._input_files
            )
            cache_dir = cache_data_path.parent
            preview_path = get_preview_downmix_path(cache_dir, cache_hash, self._preview_sr, self._preview_channels)

            if cache_data_path.exists():
                self.progress.emit("Loading cached features...")
                times, combined, rms_norm, onset_norm = load_cached_analysis(cache_data_path)

                if not preview_path.exists():
                    self.progress.emit("Creating preview downmix...")
                    create_analysis_downmix(preview_path, self._input_files, self._preview_sr, self._preview_channels)

                self.finished.emit((times, combined, rms_norm, onset_norm, preview_path))
                return

            self.progress.emit("Creating downmixes...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                analysis_downmix_path = Path(tmp.name)

            # Run analysis (2 kHz temp) and preview (configurable SR, cached) downmixes in parallel.
            with ThreadPoolExecutor(max_workers=2) as pool:
                analysis_future = pool.submit(
                    create_analysis_downmix, analysis_downmix_path, self._input_files, SR
                )
                preview_future = pool.submit(
                    create_analysis_downmix, preview_path, self._input_files, self._preview_sr, self._preview_channels
                )
                analysis_future.result()
                preview_future.result()

            self.progress.emit("Extracting audio features...")
            times, combined, rms_norm, onset_norm = extract_features(
                str(analysis_downmix_path), sr=SR, hop_length=HOP_LENGTH
            )

            self.progress.emit("Saving cache...")
            save_cached_analysis(
                    cache_data_path,
                    cache_meta_path,
                    self._input_files,
                    self._exclusions,
                    SR,
                    HOP_LENGTH,
                    times,
                    combined,
                    rms_norm,
                    onset_norm,
                )

            self.finished.emit((times, combined, rms_norm, onset_norm, preview_path))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if analysis_downmix_path is not None and analysis_downmix_path.exists():
                analysis_downmix_path.unlink()
