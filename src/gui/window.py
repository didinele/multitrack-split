import glob
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..cache import get_cache_paths, load_cached_analysis
from ..downmix import get_audio_files
from ..export import export_song, save_metadata
from ..segmentation import (
    add_padding,
    apply_hysteresis,
    extend_regions_for_ascending_start,
    get_regions,
    smooth_features,
)
from .canvas import RegionEditorCanvas
from .sidebar import SettingsSidebar
from .worker import SR, HOP_LENGTH, AnalysisWorker

_FRAMES_PER_SEC = SR // HOP_LENGTH


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("wav-split")
        self.resize(1400, 750)

        self._features: Optional[tuple] = None
        self._all_wavs: list[Path] = []
        self._output_dir: Optional[Path] = None
        self._worker: Optional[AnalysisWorker] = None
        self._thread: Optional[QThread] = None
        self._confirmed = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Splitter: sidebar | right panel ---
        splitter = QSplitter()
        root.addWidget(splitter)

        self._sidebar = SettingsSidebar()
        splitter.addWidget(self._sidebar)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._canvas = RegionEditorCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self._canvas)

        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right_layout.addWidget(self._toolbar)

        mode_bar = QHBoxLayout()
        self._edit_btn = QPushButton("Edit Regions")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setChecked(False)
        self._edit_btn.toggled.connect(self._on_mode_toggled)
        mode_bar.addWidget(self._edit_btn)
        mode_bar.addStretch()
        right_layout.addLayout(mode_bar)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- Bottom bar ---
        bottom = QHBoxLayout()
        self._region_info = QLabel("Run analysis to detect regions.")
        self._region_info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bottom.addWidget(self._region_info)
        bottom.addStretch()

        cancel_btn = QPushButton("Cancel (no export)")
        cancel_btn.clicked.connect(self.close)
        bottom.addWidget(cancel_btn)

        self._confirm_btn = QPushButton("Confirm && Export")
        self._confirm_btn.setDefault(True)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        bottom.addWidget(self._confirm_btn)

        root.addLayout(bottom)

        # --- Connections ---
        self._sidebar.run_analysis_requested.connect(self._on_run_analysis)
        self._sidebar.rerun_segmentation_requested.connect(self._on_rerun_segmentation)
        self._canvas.regions_changed.connect(self._update_region_info)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_mode_toggled(self, checked: bool):
        self._edit_btn.setText("Navigate" if checked else "Edit Regions")
        self._canvas.set_edit_mode(checked, self._toolbar)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _on_run_analysis(self):
        params = self._sidebar.get_params()
        input_dir = Path(params["input_dir"])
        output_dir = Path(params["output_dir"])

        if not input_dir.is_dir():
            self.statusBar().showMessage(f"Error: '{input_dir}' is not a valid directory.")
            return
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        exclusions = params["exclusions"]
        no_cache = params["no_cache"]

        input_files = get_audio_files(input_dir, exclusions)
        if not input_files:
            self.statusBar().showMessage("Error: No WAV files found after applying exclusions.")
            return

        self._all_wavs = [Path(f) for f in glob.glob(str(input_dir / "*.[wW][aA][vV]"))]
        self._output_dir = output_dir
        self._sidebar.set_run_enabled(False)

        cache_data_path: Optional[Path] = None
        cache_meta_path: Optional[Path] = None

        if not no_cache:
            cache_data_path, cache_meta_path, _ = get_cache_paths(input_dir, input_files)
            if cache_data_path.exists():
                self.statusBar().showMessage("Cache hit — loading features...")
                times, combined, rms_norm, onset_norm = load_cached_analysis(cache_data_path)
                self._features = (times, combined, rms_norm, onset_norm)
                self._sidebar.set_run_enabled(True)
                self._on_rerun_segmentation()
                return

        self.statusBar().showMessage("Creating analysis downmix...")
        self._worker = AnalysisWorker(
            input_files=input_files,
            exclusions=exclusions,
            cache_data_path=cache_data_path,
            cache_meta_path=cache_meta_path,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_worker_finished(self, result):
        times, combined, rms_norm, onset_norm = result
        self._features = (times, combined, rms_norm, onset_norm)
        self._sidebar.set_run_enabled(True)
        self._on_rerun_segmentation()

    def _on_worker_error(self, message: str):
        self.statusBar().showMessage(f"Error: {message}")
        self._sidebar.set_run_enabled(True)

    # ------------------------------------------------------------------
    # Segmentation (fast, runs on main thread)
    # ------------------------------------------------------------------

    def _on_rerun_segmentation(self):
        if self._features is None:
            return

        times, combined, rms_norm, onset_norm = self._features
        params = self._sidebar.get_params()

        smooth_frames = params["smooth_windows_sec"] * _FRAMES_PER_SEC
        smoothed = smooth_features(combined, smooth_frames)
        state = apply_hysteresis(smoothed, params["start_thresh"], params["stop_thresh"])
        raw_regions = get_regions(state, times, params["min_active_sec"], params["min_silence_sec"])
        extended_regions, ascension_spans = extend_regions_for_ascending_start(
            raw_regions, times, smoothed, max_lookback_sec=params["min_silence_sec"]
        )
        final_regions = add_padding(extended_regions, params["pre_pad"], params["post_pad"], float(times[-1]))

        self._canvas.update_visualization(
            times, combined, smoothed, state,
            raw_regions, extended_regions, ascension_spans, final_regions,
        )

        self._sidebar.set_rerun_enabled(True)
        self._confirm_btn.setEnabled(bool(final_regions))
        self._update_region_info()
        self.statusBar().showMessage(f"Done — {len(final_regions)} region(s) detected.")

    # ------------------------------------------------------------------
    # Region info label
    # ------------------------------------------------------------------

    def _update_region_info(self):
        regions = self._canvas.get_regions()
        if not regions:
            self._region_info.setText("No regions detected.")
            return
        parts = [
            f"Song {i + 1}: {s:.1f}s–{e:.1f}s ({e - s:.1f}s)"
            for i, (s, e) in enumerate(regions)
        ]
        self._region_info.setText("  |  ".join(parts))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_confirm(self):
        confirmed_regions = self._canvas.get_regions()
        if not confirmed_regions or self._output_dir is None:
            return

        self._confirm_btn.setEnabled(False)
        try:
            for idx, region in enumerate(confirmed_regions, 1):
                self.statusBar().showMessage(
                    f"Exporting Song {idx}/{len(confirmed_regions)}..."
                )
                QApplication.processEvents()
                export_song(idx, region, self._output_dir, self._all_wavs)

            save_metadata(confirmed_regions, self._output_dir)
            self._confirmed = True
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(confirmed_regions)} song(s) to:\n{self._output_dir}",
            )
            self.close()
        except Exception as e:
            self.statusBar().showMessage(f"Export failed: {e}")
            self._confirm_btn.setEnabled(True)
