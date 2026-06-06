import threading
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QRect, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
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
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        if self.isEnabled():
            handle_color = _COLOR_HANDLE
        else:
            handle_color = _COLOR_HANDLE_OFF
        hx = int(self._value * w)
        painter.setBrush(handle_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(hx - _HANDLE_R, mid - _HANDLE_R, _HANDLE_R * 2, _HANDLE_R * 2)

    # Mouse interaction
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

    def __init__(self, path: Path, start_time: float, end_time: Optional[float]):
        super().__init__()
        self._path = path
        self._start_time = start_time
        self._end_time = end_time
        self._stop_flag = threading.Event()

    def stop_playback(self):
        self._stop_flag.set()
        sd.stop()

    def run(self):
        done_event = threading.Event()
        # current_frame is written by the audio callback (PortAudio thread) and
        # read by the loop below (QThread). Single-element list writes are
        # effectively atomic under the GIL.
        current_frame = [0]

        try:
            with sf.SoundFile(self._path) as f:
                sr = f.samplerate
                start_frame = int(self._start_time * sr)
                end_frame = int(self._end_time * sr) if self._end_time is not None else f.frames
                end_frame = min(end_frame, f.frames)
                f.seek(start_frame)
                current_frame[0] = start_frame

                def callback(outdata, frames, time_info, status):
                    if self._stop_flag.is_set():
                        raise sd.CallbackStop()
                    remaining = end_frame - current_frame[0]
                    if remaining <= 0:
                        outdata[:] = 0
                        raise sd.CallbackStop()
                    to_read = min(frames, remaining)
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
        self._play_end_time: Optional[float] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        # --- Controls row ---
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_full_btn = QPushButton("▶ Full Track")
        self._play_full_btn.setEnabled(False)
        self._play_full_btn.clicked.connect(self._on_play_full)
        controls.addWidget(self._play_full_btn)

        self._play_region_btn = QPushButton("▶ Region")
        self._play_region_btn.setEnabled(False)
        self._play_region_btn.clicked.connect(self._on_play_region)
        controls.addWidget(self._play_region_btn)

        self._region_spin = QSpinBox()
        self._region_spin.setRange(1, 1)
        self._region_spin.setFixedWidth(48)
        self._region_spin.setVisible(False)
        controls.addWidget(self._region_spin)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop)
        controls.addWidget(self._stop_btn)

        self._refresh_btn = QPushButton("↻ Refresh")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setToolTip("Re-create preview downmix with current Preview Quality settings")
        self._refresh_btn.clicked.connect(self.refresh_preview_requested)
        controls.addWidget(self._refresh_btn)

        self._time_label = QLabel("–:– / –:–")
        controls.addWidget(self._time_label)

        controls.addStretch()
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
        self._play_full_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._seek_bar.reset()
        self._seek_bar.set_regions(self._regions, total_duration)
        self._time_label.setText(f"0:00 / {_fmt_time(total_duration)}")
        self._sync_region_btn()

    def set_refreshing(self, refreshing: bool):
        self._refresh_btn.setEnabled(not refreshing and self._preview_path is not None)

    def update_regions(self, regions: list[tuple[float, float]]):
        self._regions = regions
        self._region_spin.setVisible(bool(regions))
        if regions:
            self._region_spin.setRange(1, len(regions))
        self._seek_bar.set_regions(regions, self._total_duration)
        self._sync_region_btn()

    def is_playing(self) -> bool:
        return self._worker is not None

    def skip(self, seconds: float):
        if self._worker is None or self._total_duration <= 0:
            return
        current = self._seek_bar._value * self._total_duration
        new_time = max(0.0, current + seconds)
        if self._play_end_time is not None:
            new_time = min(new_time, self._play_end_time - 0.1)
        self._start_playback(new_time, self._play_end_time)

    def stop(self):
        if self._worker is not None:
            # Disconnect before stopping so that a stale `finished` signal queued
            # during worker.wait() cannot fire after a new worker has been assigned.
            try:
                self._worker.position_changed.disconnect(self._on_position)
                self._worker.finished.disconnect(self._on_worker_finished)
            except RuntimeError:
                pass
            self._worker.stop_playback()
            self._worker.wait(500)
            self._worker = None
        self._set_playing(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_region_btn(self):
        self._play_region_btn.setEnabled(
            self._preview_path is not None and bool(self._regions) and self._worker is None
        )

    def _on_play_full(self):
        if self._preview_path is None:
            return
        self._seek_bar.set_active_region(None)
        self._start_playback(0.0, None)

    def _on_play_region(self):
        if self._preview_path is None or not self._regions:
            return
        idx = self._region_spin.value() - 1
        self._seek_bar.set_active_region(idx)
        start, end = self._regions[idx]
        self._start_playback(start, end)

    def _start_playback(self, start_time: float, end_time: Optional[float]):
        self.stop()
        self._play_end_time = end_time
        self._worker = _PlaybackWorker(self._preview_path, start_time, end_time)
        self._worker.position_changed.connect(self._on_position)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self._set_playing(True)

    @Slot()
    def _on_worker_finished(self):
        self._set_playing(False)
        self._worker = None

    def _set_playing(self, playing: bool):
        self._play_full_btn.setEnabled(not playing and self._preview_path is not None)
        self._play_region_btn.setEnabled(
            not playing and self._preview_path is not None and bool(self._regions)
        )
        self._stop_btn.setEnabled(playing)
        self._seek_bar.setEnabled(playing)
        if not playing:
            self._time_label.setText(f"0:00 / {_fmt_time(self._total_duration)}")
            self._seek_bar.set_active_region(None)

    @Slot(float)
    def _on_position(self, pos: float):
        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(self._total_duration)}")
        if self._total_duration > 0:
            self._seek_bar.set_value(pos / self._total_duration)

    def _on_seeked(self, normalized: float):
        if self._worker is None:
            return
        seek_time = normalized * self._total_duration
        end_time = self._play_end_time
        if end_time is not None:
            seek_time = min(seek_time, end_time - 0.1)
        # Preserve the active region highlight across the seek restart.
        active = self._seek_bar._active_region_idx
        self._start_playback(seek_time, end_time)
        self._seek_bar.set_active_region(active)
