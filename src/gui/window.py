import glob
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, Qt, QThread
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QFrame,
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..downmix import get_audio_files
from ..export import export_song
from ..segmentation import (
    add_padding,
    apply_hysteresis,
    extend_regions_for_ascending_start,
    get_regions,
    smooth_features,
)
from .canvas import WaveformCanvas
from .player import PreviewPlayer
from .preview_worker import PreviewCreationWorker
from .sidebar import SettingsSidebar
from .worker import SR, HOP_LENGTH, AnalysisWorker

_FRAMES_PER_SEC = SR // HOP_LENGTH

def _fmt_time(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("multitrack-split")
        self.resize(1100, 620)

        self._features: Optional[tuple] = None
        self._all_wavs: list[Path] = []
        self._output_dir: Optional[Path] = None
        self._worker: Optional[AnalysisWorker] = None
        self._thread: Optional[QThread] = None
        self._confirmed = False
        self._preview_worker: Optional[PreviewCreationWorker] = None
        self._preview_thread: Optional[QThread] = None

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

        self._canvas = WaveformCanvas()
        right_layout.addWidget(self._canvas)

        self._player = PreviewPlayer()
        right_layout.addWidget(self._player)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])

        # --- Bottom bar ---
        bottom = QHBoxLayout()
        self._region_info = QLabel("")
        info_scroll = QScrollArea()
        info_scroll.setWidget(self._region_info)
        info_scroll.setWidgetResizable(False)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        info_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        info_scroll.setFixedHeight(24)
        bottom.addWidget(info_scroll)

        cancel_btn = QPushButton("Cancel")
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
        self._canvas.regions_changed.connect(self._on_canvas_regions_changed)
        self._player.refresh_preview_requested.connect(self._on_refresh_preview)

        QApplication.instance().installEventFilter(self)

        # --- Status bar progress indicator ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setFixedWidth(180)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

    def closeEvent(self, event):
        QApplication.instance().removeEventFilter(self)
        self._sidebar.save_settings()
        self._player.stop()
        for thread, worker in [
            (self._thread, self._worker),
            (self._preview_thread, self._preview_worker),
        ]:
            if thread is not None:
                try:
                    worker.progress.disconnect()
                    worker.finished.disconnect()
                    worker.error.disconnect()
                except RuntimeError:
                    pass
                thread.quit()
                thread.wait()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            focused = QApplication.focusWidget()
            # Don't steal arrow keys from text-entry widgets (QLineEdit covers
            # the internal editors inside QSpinBox / QDoubleSpinBox / QComboBox).
            if not isinstance(focused, (QLineEdit, QAbstractSpinBox)):
                if event.key() == Qt.Key.Key_Left and self._player.is_playing():
                    self._player.skip(-5.0)
                    return True
                if event.key() == Qt.Key.Key_Right and self._player.is_playing():
                    self._player.skip(5.0)
                    return True
        return False

    def _set_loading(self, loading: bool):
        self._progress_bar.setVisible(loading)
        self._sidebar.set_run_enabled(not loading)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

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

        input_files = get_audio_files(input_dir, exclusions)
        if not input_files:
            self.statusBar().showMessage("Error: No WAV files found after applying exclusions.")
            return

        self._all_wavs = [Path(f) for f in glob.glob(str(input_dir / "*.[wW][aA][vV]"))]
        self._output_dir = output_dir
        self._player.stop()
        self._set_loading(True)
        self.statusBar().showMessage("Starting analysis...")

        self._worker = AnalysisWorker(
            input_files=input_files,
            input_dir=input_dir,
            exclusions=exclusions,
            preview_sr=params["preview_sr"],
            preview_channels=params["preview_channels"],
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_analysis_thread_done)

        self._thread.start()

    def _on_worker_finished(self, result):
        times, combined, rms_norm, onset_norm, preview_path = result
        self._features = (times, combined, rms_norm, onset_norm)
        self._player.set_preview(preview_path, float(times[-1]))
        self._set_loading(False)
        self._on_rerun_segmentation()

    def _on_worker_error(self, message: str):
        self._set_loading(False)
        self.statusBar().showMessage(f"Error: {message}")

    def _on_analysis_thread_done(self):
        self._worker = None
        self._thread = None

    def _on_preview_thread_done(self):
        self._preview_worker = None
        self._preview_thread = None

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

        self._canvas.update_visualization(times, combined, final_regions)
        self._player.update_regions(final_regions)

        self._sidebar.set_rerun_enabled(True)
        self._confirm_btn.setEnabled(bool(final_regions))
        self._update_region_info()
        self.statusBar().showMessage(f"Done — {len(final_regions)} region(s) detected.")

    # ------------------------------------------------------------------
    # Refresh preview
    # ------------------------------------------------------------------

    def _on_refresh_preview(self):
        params = self._sidebar.get_params()
        input_dir = Path(params["input_dir"])
        if not input_dir.is_dir() or not self._all_wavs:
            self.statusBar().showMessage("Error: run analysis first before refreshing preview.")
            return

        exclusions = params["exclusions"]
        input_files = get_audio_files(input_dir, exclusions)
        if not input_files:
            return

        self._player.stop()
        self._player.set_refreshing(True)
        self._set_loading(True)
        self.statusBar().showMessage("Refreshing preview downmix...")

        self._preview_worker = PreviewCreationWorker(
            input_files=input_files,
            input_dir=input_dir,
            preview_sr=params["preview_sr"],
            preview_channels=params["preview_channels"],
        )
        self._preview_thread = QThread()
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.progress.connect(self.statusBar().showMessage)
        self._preview_worker.finished.connect(self._on_preview_worker_finished)
        self._preview_worker.error.connect(self._on_worker_error)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.error.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._on_preview_thread_done)

        self._preview_thread.start()

    def _on_preview_worker_finished(self, preview_path):
        if self._features is not None:
            times = self._features[0]
            self._player.set_preview(preview_path, float(times[-1]))
        else:
            self._player.set_refreshing(False)
        self._set_loading(False)
        self.statusBar().showMessage("Preview refreshed.")

    # ------------------------------------------------------------------
    # Region info label
    # ------------------------------------------------------------------

    def _on_canvas_regions_changed(self):
        self._player.update_regions(self._canvas.get_regions())

    def _update_region_info(self):
        regions = self._canvas.get_regions()
        if not regions:
            self._region_info.setText("")
            self._region_info.adjustSize()
            return
        parts = [
            f"Song {i + 1}: {_fmt_time(s)} – {_fmt_time(e)}  ({_fmt_time(e - s)})"
            for i, (s, e) in enumerate(regions)
        ]
        self._region_info.setText("  |  ".join(parts))
        self._region_info.adjustSize()

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
