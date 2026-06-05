import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..downmix import create_analysis_downmix
from ..features import extract_features
from ..cache import save_cached_analysis

SR = 2000
HOP_LENGTH = 200


class AnalysisWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        input_files: list[Path],
        exclusions: list[str],
        cache_data_path: "Path | None",
        cache_meta_path: "Path | None",
    ):
        super().__init__()
        self._input_files = input_files
        self._exclusions = exclusions
        self._cache_data_path = cache_data_path
        self._cache_meta_path = cache_meta_path

    @Slot()
    def run(self):
        downmix_path = None
        try:
            self.progress.emit("Creating analysis downmix...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                downmix_path = Path(tmp.name)

            create_analysis_downmix(downmix_path, self._input_files, SR)

            self.progress.emit("Extracting audio features...")
            times, combined, rms_norm, onset_norm = extract_features(
                str(downmix_path), sr=SR, hop_length=HOP_LENGTH
            )

            if self._cache_data_path is not None:
                self.progress.emit("Saving cache...")
                save_cached_analysis(
                    self._cache_data_path,
                    self._cache_meta_path,
                    self._input_files,
                    self._exclusions,
                    SR,
                    HOP_LENGTH,
                    times,
                    combined,
                    rms_norm,
                    onset_norm,
                )

            self.finished.emit((times, combined, rms_norm, onset_norm))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if downmix_path is not None and downmix_path.exists():
                downmix_path.unlink()
