import threading
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QRect, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _fmt_time(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ------------------------------------------------------------------
# Custom seek bar with region overlays
# ------------------------------------------------------------------

_TRACK_H = 6
_HANDLE_R = 7

_COLOR_TRACK = QColor(50, 50, 50)
_COLOR_REGION = QColor(80, 120, 180, 90)
_COLOR_REGION_ACTIVE = QColor(80, 160, 240, 170)
_COLOR_HANDLE = QColor(220, 220, 220)
_COLOR_HANDLE_OFF = QColor(90, 90, 90)


class RegionSeekBar(QWidget):
    seeked = Signal(float)  # emits normalised value 0.0–1.0 on mouse release

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(_HANDLE_R * 2 + 6)
        # ClickFocus so clicking the seek bar steals keyboard focus from the
        # region spinbox (and any other widget that might hold it).
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value: float = 0.0
        self._regions: list[tuple[float, float]] = []
        self._total_duration: float = 0.0
        self._active_region_idx: Optional[int] = None
        self._dragging: bool = False

    # Public API
    def set_value(self, v: float):
        if self._dragging:
            return
        self._value = max(0.0, min(1.0, v))
        self.update()

    def seek_to(self, v: float):
        """Force-set the position even while the user is not dragging (e.g. from region selector)."""
        self._value = max(0.0, min(1.0, v))
        self.update()

    def reset(self):
        self._value = 0.0
        self._active_region_idx = None
        self.update()

    def set_regions(self, regions: list[tuple[float, float]], total_duration: float):
        self._regions = regions
        self._total_duration = total_duration
        self.update()

    def set_active_region(self, idx: Optional[int]):
        self._active_region_idx = idx
        self.update()

    # Paint
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        mid = h // 2
        track_y = mid - _TRACK_H // 2

        # Background track
        painter.fillRect(QRect(0, track_y, w, _TRACK_H), _COLOR_TRACK)

        # Region overlays
        if self._regions and self._total_duration > 0:
            for i, (start, end) in enumerate(self._regions):
                x1 = int(start / self._total_duration * w)
                x2 = int(end / self._total_duration * w)
                color = _COLOR_REGION_ACTIVE if i == self._active_region_idx else _COLOR_REGION
                painter.fillRect(QRect(x1, track_y, max(1, x2 - x1), _TRACK_H), color)

        # Handle
        handle_color = _COLOR_HANDLE if self.isEnabled() else _COLOR_HANDLE_OFF
        hx = int(self._value * w)
        painter.setBrush(handle_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(hx - _HANDLE_R, mid - _HANDLE_R, _HANDLE_R * 2, _HANDLE_R * 2)

    # Mouse interaction — always interactable when enabled, no matter play state
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._dragging = True
            self._set_from_x(event.position().x())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._set_from_x(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._set_from_x(event.position().x())
            self.seeked.emit(self._value)

    def _set_from_x(self, x: float):
        self._value = max(0.0, min(1.0, x / max(1, self.width())))
        self.update()


# ------------------------------------------------------------------
# Playback worker
# ------------------------------------------------------------------

class _PlaybackWorker(QThread):
    position_changed = Signal(float)
    finished = Signal()

    def __init__(self, path: Path, start_time: float):
        super().__init__()
        self._path = path
        self._start_time = start_time
        self._stop_flag = threading.Event()

    def stop_playback(self):
        self._stop_flag.set()

    def run(self):
        done_event = threading.Event()
        current_frame = [0]

        try:
            with sf.SoundFile(self._path) as f:
                sr = f.samplerate
                start_frame = int(self._start_time * sr)
                f.seek(start_frame)
                current_frame[0] = start_frame

                def callback(outdata, frames, time_info, status):
                    if self._stop_flag.is_set():
                        raise sd.CallbackStop()
                    to_read = min(frames, f.frames - current_frame[0])
                    if to_read <= 0:
                        outdata[:] = 0
                        raise sd.CallbackStop()
                    data = f.read(to_read, dtype="float32", always_2d=True)
                    actual = len(data)
                    outdata[:actual] = data
                    if actual < frames:
                        outdata[actual:] = 0
                    current_frame[0] += actual
                    if actual < frames:
                        raise sd.CallbackStop()

                stream = sd.OutputStream(
                    samplerate=sr,
                    channels=f.channels,
                    callback=callback,
                    finished_callback=done_event.set,
                )
                with stream:
                    while not done_event.wait(0.1):
                        if self._stop_flag.is_set():
                            # Wait for the finished_callback before exiting the context
                            # manager so the stream closes cleanly (avoids AUHAL -50).
                            done_event.wait(2.0)
                            break
                        self.position_changed.emit(current_frame[0] / sr)

        except Exception:
            pass
        finally:
            self.finished.emit()


# ------------------------------------------------------------------
# Player widget
# ------------------------------------------------------------------

class PreviewPlayer(QWidget):
    refresh_preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._preview_path: Optional[Path] = None
        self._total_duration: float = 0.0
        self._regions: list[tuple[float, float]] = []
        self._worker: Optional[_PlaybackWorker] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        # --- Controls row ---
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_pause_btn = QPushButton("▶ Play")
        self._play_pause_btn.setEnabled(False)
        self._play_pause_btn.clicked.connect(self._on_play_pause)
        controls.addWidget(self._play_pause_btn)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop)
        controls.addWidget(self._stop_btn)

        self._refresh_btn = QPushButton("↻ Refresh")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setToolTip("Re-create preview downmix with current Preview Quality settings")
        self._refresh_btn.clicked.connect(self.refresh_preview_requested)
        controls.addWidget(self._refresh_btn)

        self._region_label = QLabel("Region:")
        self._region_label.setVisible(False)
        controls.addWidget(self._region_label)

        self._region_combo = QComboBox()
        self._region_combo.setVisible(False)
        self._region_combo.setToolTip("Jump to the start of this region")
        self._region_combo.setMinimumWidth(120)
        self._region_combo.currentIndexChanged.connect(self._on_region_changed)
        controls.addWidget(self._region_combo)

        controls.addStretch()

        self._time_label = QLabel("–:– / –:–")
        controls.addWidget(self._time_label)

        outer.addLayout(controls)

        # --- Seek bar with region overlays ---
        self._seek_bar = RegionSeekBar()
        self._seek_bar.setEnabled(False)
        self._seek_bar.seeked.connect(self._on_seeked)
        outer.addWidget(self._seek_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_preview(self, path: Path, total_duration: float):
        self.stop()
        self._preview_path = path
        self._total_duration = total_duration
        self._seek_bar.reset()
        self._seek_bar.set_regions(self._regions, total_duration)
        self._update_controls()

    def set_refreshing(self, refreshing: bool):
        self._refresh_btn.setEnabled(not refreshing and self._preview_path is not None)

    def update_regions(self, regions: list[tuple[float, float]]):
        self._regions = regions
        has = bool(regions)
        self._region_label.setVisible(has)
        self._region_combo.setVisible(has)
        if has:
            self._region_combo.blockSignals(True)
            self._region_combo.clear()
            for i, (start, _) in enumerate(regions):
                self._region_combo.addItem(f"Song {i + 1}  —  {_fmt_time(start)}")
            self._region_combo.blockSignals(False)
        self._seek_bar.set_regions(regions, self._total_duration)

    def is_playing(self) -> bool:
        return self._worker is not None

    def skip(self, seconds: float):
        if self._total_duration <= 0:
            return
        current = self._seek_bar._value * self._total_duration
        new_time = max(0.0, min(self._total_duration, current + seconds))
        self._seek_bar.seek_to(new_time / self._total_duration)
        if self._worker is not None:
            self._start_playback(new_time)

    def stop(self):
        self._halt_worker()
        self._seek_bar.reset()
        self._update_controls()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _halt_worker(self):
        """Stop the worker without touching the seek bar position (used for pause)."""
        if self._worker is not None:
            try:
                self._worker.position_changed.disconnect(self._on_position)
                self._worker.finished.disconnect(self._on_worker_finished)
            except RuntimeError:
                pass
            self._worker.stop_playback()
            self._worker.wait(500)
            self._worker = None

    def _on_play_pause(self):
        if self._worker is not None:
            # Pause: stop worker but keep the seek bar position for resume.
            self._halt_worker()
            self._update_controls()
        else:
            # Play from current seek-bar position.
            start = self._seek_bar._value * self._total_duration
            self._start_playback(start)

    def _start_playback(self, start_time: float):
        self._halt_worker()
        self._worker = _PlaybackWorker(self._preview_path, start_time)
        self._worker.position_changed.connect(self._on_position)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self._update_controls()

    @Slot()
    def _on_worker_finished(self):
        self._worker = None
        self._update_controls()

    def _update_controls(self):
        has_preview = self._preview_path is not None
        is_playing = self._worker is not None
        not_at_start = self._seek_bar._value > 0.0

        self._play_pause_btn.setText("⏸ Pause" if is_playing else "▶ Play")
        self._play_pause_btn.setEnabled(has_preview)
        self._stop_btn.setEnabled(has_preview and (is_playing or not_at_start))
        self._refresh_btn.setEnabled(has_preview)
        self._seek_bar.setEnabled(has_preview)
        if has_preview:
            self._time_label.setText(
                f"0:00 / {_fmt_time(self._total_duration)}" if not is_playing
                else self._time_label.text()
            )

    @Slot(float)
    def _on_position(self, pos: float):
        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(self._total_duration)}")
        if self._total_duration > 0:
            self._seek_bar.set_value(pos / self._total_duration)

    def _on_seeked(self, normalized: float):
        if self._worker is not None:
            self._start_playback(normalized * self._total_duration)

    def _on_region_changed(self, idx: int):
        if not self._regions or self._preview_path is None or self._total_duration <= 0:
            return
        if idx < 0 or idx >= len(self._regions):
            return
        start, _ = self._regions[idx]
        normalized = start / self._total_duration
        self._seek_bar.seek_to(normalized)
        self._seek_bar.set_active_region(idx)
        if self._worker is not None:
            self._start_playback(start)
