from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..downmix import create_analysis_downmix
from ..cache import get_cache_paths, get_preview_downmix_path


class PreviewCreationWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)  # Path to the new preview WAV
    error = Signal(str)

    def __init__(
        self,
        input_files: list[Path],
        input_dir: Path,
        preview_sr: int,
        preview_channels: int,
    ):
        super().__init__()
        self._input_files = input_files
        self._input_dir = input_dir
        self._preview_sr = preview_sr
        self._preview_channels = preview_channels

    @Slot()
    def run(self):
        try:
            self.progress.emit("Computing cache hash...")
            cache_data_path, _, cache_hash = get_cache_paths(self._input_dir, self._input_files)
            cache_dir = cache_data_path.parent
            preview_path = get_preview_downmix_path(cache_dir, cache_hash, self._preview_sr, self._preview_channels)

            self.progress.emit("Creating preview downmix...")
            create_analysis_downmix(preview_path, self._input_files, self._preview_sr, self._preview_channels)

            self.finished.emit(preview_path)
        except Exception as e:
            self.error.emit(str(e))
