from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

class SettingsSidebar(QWidget):
    run_analysis_requested = Signal()
    rerun_segmentation_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # --- Directories ---
        self._input_dir = QLineEdit()
        self._input_dir.setPlaceholderText("Select folder...")
        form.addRow("Input:", self._make_dir_row(self._input_dir))

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Select folder...")
        form.addRow("Output:", self._make_dir_row(self._output_dir))

        self._exclusions = QLineEdit()
        self._exclusions.setPlaceholderText("click")
        self._exclusions.setToolTip(
            "Comma-separated keywords; files whose name contains any of these are excluded from analysis"
        )
        form.addRow("Exclusions:", self._exclusions)

        form.addRow(self._separator())

        # --- Segmentation parameters ---
        form.addRow(QLabel("<b>Segmentation</b>"))

        self._start_thresh = QDoubleSpinBox()
        self._start_thresh.setRange(0.0, 1.0)
        self._start_thresh.setSingleStep(0.05)
        self._start_thresh.setDecimals(2)
        self._start_thresh.setValue(0.3)
        self._start_thresh.setToolTip("Signal must rise above this to start a region")
        form.addRow("Start thresh:", self._start_thresh)

        self._stop_thresh = QDoubleSpinBox()
        self._stop_thresh.setRange(0.0, 1.0)
        self._stop_thresh.setSingleStep(0.05)
        self._stop_thresh.setDecimals(2)
        self._stop_thresh.setValue(0.1)
        self._stop_thresh.setToolTip("Signal must fall below this to end a region")
        form.addRow("Stop thresh:", self._stop_thresh)

        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(1, 300)
        self._smooth_window.setValue(10)
        self._smooth_window.setSuffix(" s")
        self._smooth_window.setToolTip("Moving-average window applied to features before thresholding")
        form.addRow("Smooth window:", self._smooth_window)

        self._min_active = QSpinBox()
        self._min_active.setRange(1, 3600)
        self._min_active.setValue(60)
        self._min_active.setSuffix(" s")
        self._min_active.setToolTip("Minimum duration for a detected region to count as a song")
        form.addRow("Min active:", self._min_active)

        self._min_silence = QSpinBox()
        self._min_silence.setRange(1, 3600)
        self._min_silence.setValue(30)
        self._min_silence.setSuffix(" s")
        self._min_silence.setToolTip("Minimum silence gap between two separate songs")
        form.addRow("Min silence:", self._min_silence)

        form.addRow(self._separator())
        form.addRow(QLabel("<b>Padding</b>"))

        self._pre_pad = QSpinBox()
        self._pre_pad.setRange(0, 300)
        self._pre_pad.setValue(25)
        self._pre_pad.setSuffix(" s")
        self._pre_pad.setToolTip("Extra seconds added before each detected region")
        form.addRow("Pre-pad:", self._pre_pad)

        self._post_pad = QSpinBox()
        self._post_pad.setRange(0, 300)
        self._post_pad.setValue(25)
        self._post_pad.setSuffix(" s")
        self._post_pad.setToolTip("Extra seconds added after each detected region")
        form.addRow("Post-pad:", self._post_pad)

        form.addRow(self._separator())

        self._no_cache = QCheckBox("Ignore cache")
        self._no_cache.setToolTip("Force re-computation even if a cached analysis exists")
        form.addRow(self._no_cache)

        form.addRow(self._separator())

        self._rerun_btn = QPushButton("Re-run Segmentation")
        self._rerun_btn.setEnabled(False)
        self._rerun_btn.setToolTip("Re-apply segmentation heuristics using current parameters (fast — no re-analysis needed)")
        form.addRow(self._rerun_btn)
        self._rerun_btn.clicked.connect(self.rerun_segmentation_requested)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setToolTip("Re-analyse only when input files have changed — use Re-run Segmentation to adjust parameters")
        form.addRow(self._run_btn)
        self._run_btn.clicked.connect(self.run_analysis_requested)

        restore_btn = QPushButton("Restore Defaults")
        restore_btn.setToolTip("Reset all parameters to their default values (directories are unchanged)")
        form.addRow(restore_btn)
        restore_btn.clicked.connect(self._restore_defaults)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        outer = QFormLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addRow(scroll)

        self._load_settings()

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(line_edit)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(lambda: self._pick_dir(line_edit))
        row.addWidget(btn)
        return container

    def _pick_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)

    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def get_params(self) -> dict:
        raw_excl = [x.strip() for x in self._exclusions.text().split(",") if x.strip()]
        return {
            "input_dir": self._input_dir.text().strip(),
            "output_dir": self._output_dir.text().strip(),
            "exclusions": raw_excl if raw_excl else ["click"],
            "start_thresh": self._start_thresh.value(),
            "stop_thresh": self._stop_thresh.value(),
            "smooth_windows_sec": self._smooth_window.value(),
            "min_active_sec": self._min_active.value(),
            "min_silence_sec": self._min_silence.value(),
            "pre_pad": self._pre_pad.value(),
            "post_pad": self._post_pad.value(),
            "no_cache": self._no_cache.isChecked(),
        }

    def set_run_enabled(self, enabled: bool):
        self._run_btn.setEnabled(enabled)

    def set_rerun_enabled(self, enabled: bool):
        self._rerun_btn.setEnabled(enabled)

    def _restore_defaults(self):
        self._exclusions.clear()
        self._start_thresh.setValue(0.3)
        self._stop_thresh.setValue(0.1)
        self._smooth_window.setValue(10)
        self._min_active.setValue(60)
        self._min_silence.setValue(30)
        self._pre_pad.setValue(25)
        self._post_pad.setValue(25)
        self._no_cache.setChecked(False)

    def _load_settings(self):
        s = QSettings("wav-split", "wav-split")
        if s.contains("input_dir"):
            self._input_dir.setText(s.value("input_dir", ""))
        if s.contains("output_dir"):
            self._output_dir.setText(s.value("output_dir", ""))
        if s.contains("exclusions"):
            self._exclusions.setText(s.value("exclusions", ""))
        if s.contains("start_thresh"):
            self._start_thresh.setValue(float(s.value("start_thresh", 0.3)))
        if s.contains("stop_thresh"):
            self._stop_thresh.setValue(float(s.value("stop_thresh", 0.1)))
        if s.contains("smooth_windows_sec"):
            self._smooth_window.setValue(int(s.value("smooth_windows_sec", 10)))
        if s.contains("min_active_sec"):
            self._min_active.setValue(int(s.value("min_active_sec", 60)))
        if s.contains("min_silence_sec"):
            self._min_silence.setValue(int(s.value("min_silence_sec", 30)))
        if s.contains("pre_pad"):
            self._pre_pad.setValue(int(s.value("pre_pad", 25)))
        if s.contains("post_pad"):
            self._post_pad.setValue(int(s.value("post_pad", 25)))
        if s.contains("no_cache"):
            self._no_cache.setChecked(s.value("no_cache", False, type=bool))

    def save_settings(self):
        params = self.get_params()
        s = QSettings("wav-split", "wav-split")
        s.setValue("input_dir", params["input_dir"])
        s.setValue("output_dir", params["output_dir"])
        s.setValue("exclusions", self._exclusions.text())
        s.setValue("start_thresh", params["start_thresh"])
        s.setValue("stop_thresh", params["stop_thresh"])
        s.setValue("smooth_windows_sec", params["smooth_windows_sec"])
        s.setValue("min_active_sec", params["min_active_sec"])
        s.setValue("min_silence_sec", params["min_silence_sec"])
        s.setValue("pre_pad", params["pre_pad"])
        s.setValue("post_pad", params["post_pad"])
        s.setValue("no_cache", params["no_cache"])
