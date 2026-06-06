import threading
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QWidget,
)


def _fmt_time(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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


class PreviewPlayer(QWidget):
    refresh_preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._preview_path: Optional[Path] = None
        self._total_duration: float = 0.0
        self._regions: list[tuple[float, float]] = []
        self._worker: Optional[_PlaybackWorker] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self._play_full_btn = QPushButton("▶ Full Track")
        self._play_full_btn.setEnabled(False)
        self._play_full_btn.clicked.connect(self._on_play_full)
        layout.addWidget(self._play_full_btn)

        self._play_region_btn = QPushButton("▶ Region")
        self._play_region_btn.setEnabled(False)
        self._play_region_btn.clicked.connect(self._on_play_region)
        layout.addWidget(self._play_region_btn)

        self._region_spin = QSpinBox()
        self._region_spin.setRange(1, 1)
        self._region_spin.setFixedWidth(48)
        self._region_spin.setVisible(False)
        layout.addWidget(self._region_spin)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop)
        layout.addWidget(self._stop_btn)

        self._refresh_btn = QPushButton("↻ Refresh")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setToolTip("Re-create preview downmix with current Preview Quality settings")
        self._refresh_btn.clicked.connect(self.refresh_preview_requested)
        layout.addWidget(self._refresh_btn)

        self._time_label = QLabel("–:– / –:–")
        layout.addWidget(self._time_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_preview(self, path: Path, total_duration: float):
        self.stop()
        self._preview_path = path
        self._total_duration = total_duration
        self._play_full_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._time_label.setText(f"0:00 / {_fmt_time(total_duration)}")
        # Re-apply region button state in case regions were set before preview arrived.
        self._sync_region_btn()

    def set_refreshing(self, refreshing: bool):
        self._refresh_btn.setEnabled(not refreshing and self._preview_path is not None)

    def update_regions(self, regions: list[tuple[float, float]]):
        self._regions = regions
        self._region_spin.setVisible(bool(regions))
        if regions:
            self._region_spin.setRange(1, len(regions))
        self._sync_region_btn()

    def stop(self):
        if self._worker is not None:
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
        self._start_playback(0.0, None)

    def _on_play_region(self):
        if self._preview_path is None or not self._regions:
            return
        idx = self._region_spin.value() - 1
        start, end = self._regions[idx]
        self._start_playback(start, end)

    def _start_playback(self, start_time: float, end_time: Optional[float]):
        self.stop()
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
        if not playing:
            self._time_label.setText(f"0:00 / {_fmt_time(self._total_duration)}")

    @Slot(float)
    def _on_position(self, pos: float):
        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(self._total_duration)}")
