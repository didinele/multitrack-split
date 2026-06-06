from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

DRAG_THRESHOLD_PX = 8
EPSILON = 1.0
_PAD_TOP = 8
_PAD_BOTTOM = 28

_PALETTE = np.array([
    [30,  30,  30,  255],  # 0 — background
    [90,  90,  90,  255],  # 1 — inactive bar
    [80,  160, 240, 255],  # 2 — active (in-region) bar
], dtype=np.uint8)

_BG = QColor(30, 30, 30)
_HANDLE = QColor(220, 60, 60)
_AXIS_FG = QColor(150, 150, 150)


def _fmt_axis(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class DragState:
    region_idx: int
    which: str  # "start" or "end"
    t_min: float
    t_max: float


class WaveformCanvas(QWidget):
    regions_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._times: Optional[np.ndarray] = None
        self._combined: Optional[np.ndarray] = None
        self._regions: list[list[float]] = []
        self._drag: Optional[DragState] = None

        # Caches — invalidated on resize or data change, reused across drags
        self._bucket_cache: Optional[np.ndarray] = None
        self._bucket_cache_w: int = 0
        self._active_cache: Optional[np.ndarray] = None
        self._active_cache_sz: tuple[int, int] = (0, 0)

        self.setMinimumSize(200, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_visualization(self, times: np.ndarray, combined: np.ndarray, regions: list):
        self._times = times
        self._combined = combined
        self._regions = [list(r) for r in regions]
        self._invalidate_waveform_cache()
        self.update()

    def set_regions(self, regions: list):
        self._regions = [list(r) for r in regions]
        self.update()

    def get_regions(self) -> list[tuple[float, float]]:
        return [(r[0], r[1]) for r in self._regions]

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _t_to_x(self, t: float) -> float:
        t0, t1 = float(self._times[0]), float(self._times[-1])
        return (t - t0) / (t1 - t0) * self.width()

    def _x_to_t(self, x: float) -> float:
        t0, t1 = float(self._times[0]), float(self._times[-1])
        return t0 + (x / self.width()) * (t1 - t0)

    # ------------------------------------------------------------------
    # Waveform caches
    # ------------------------------------------------------------------

    def _invalidate_waveform_cache(self):
        self._bucket_cache = None
        self._bucket_cache_w = 0
        self._active_cache = None
        self._active_cache_sz = (0, 0)

    def _get_buckets(self, w: int) -> np.ndarray:
        if self._bucket_cache is not None and self._bucket_cache_w == w:
            return self._bucket_cache

        data = self._combined
        if data is None or len(data) == 0:
            result = np.zeros(w)
        else:
            n = len(data)
            edges = np.linspace(0, n, w + 1).astype(int)
            starts = edges[:-1]
            unique = np.concatenate(([True], starts[1:] > starts[:-1]))
            if unique.all():
                result = np.maximum.reduceat(data, starts).astype(float)
            else:
                result = np.interp(np.linspace(0, n - 1, w), np.arange(n), data)
            mx = result.max()
            if mx > 0:
                result /= mx

        self._bucket_cache = result
        self._bucket_cache_w = w
        return result

    def _get_active(self, w: int, wave_h: int) -> np.ndarray:
        """(wave_h, w) bool — True where a waveform bar pixel should be drawn."""
        if self._active_cache is not None and self._active_cache_sz == (w, wave_h):
            return self._active_cache

        buckets = self._get_buckets(w)
        bar_h = np.maximum(1, (buckets * wave_h).astype(int))
        y = np.arange(wave_h)[:, np.newaxis]          # (H, 1)
        thresh = (wave_h - bar_h)[np.newaxis, :]      # (1, W)
        active = y >= thresh                           # (H, W)

        self._active_cache = active
        self._active_cache_sz = (w, wave_h)
        return active

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        wave_h = max(1, h - _PAD_TOP - _PAD_BOTTOM)

        painter.fillRect(0, 0, w, h, _BG)

        if self._times is None:
            painter.end()
            return

        # Region mask: which x-columns are inside a confirmed region
        region_mask = np.zeros(w, dtype=bool)
        for r0, r1 in self._regions:
            x0 = max(0, int(self._t_to_x(r0)))
            x1 = min(w, int(self._t_to_x(r1)))
            region_mask[x0:x1] = True

        # Build category array then map through the palette LUT
        active = self._get_active(w, wave_h)           # (H, W) bool — cached
        rm = region_mask[np.newaxis, :]                # (1, W)

        category = np.zeros((wave_h, w), dtype=np.uint8)
        category[active & ~rm] = 1
        category[active & rm] = 2

        img_arr = _PALETTE[category]                   # (H, W, 4) — C-contiguous
        img = QImage(img_arr.data, w, wave_h, w * 4, QImage.Format.Format_RGBA8888)
        painter.drawImage(0, _PAD_TOP, img)

        # Region boundary handles
        pen = QPen(_HANDLE)
        pen.setWidth(2)
        painter.setPen(pen)
        for r0, r1 in self._regions:
            x0 = round(self._t_to_x(r0))
            x1 = round(self._t_to_x(r1))
            painter.drawLine(x0, _PAD_TOP, x0, _PAD_TOP + wave_h)
            painter.drawLine(x1, _PAD_TOP, x1, _PAD_TOP + wave_h)

        self._draw_time_axis(painter, w, h, _PAD_TOP + wave_h)
        painter.end()

    def _draw_time_axis(self, painter: QPainter, w: int, h: int, y_base: int):
        t0, t1 = float(self._times[0]), float(self._times[-1])
        duration = t1 - t0
        target = max(4, w // 120)
        raw = duration / target
        interval = 1
        for iv in [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]:
            interval = iv
            if iv >= raw:
                break

        painter.setPen(QPen(_AXIS_FG))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        t = (int(t0 / interval) + 1) * interval
        while t <= t1:
            x = round(self._t_to_x(t))
            painter.drawLine(x, y_base, x, y_base + 4)
            painter.drawText(x - 22, y_base + 6, 44, 18,
                             Qt.AlignmentFlag.AlignCenter, _fmt_axis(t))
            t += interval

    # ------------------------------------------------------------------
    # Resize — invalidate bucket/active caches
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        self._invalidate_waveform_cache()
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Mouse events — drag handles
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._times is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        px = event.position().x()
        best_dist = DRAG_THRESHOLD_PX + 1
        best_idx, best_which = -1, ""

        for i, (r0, r1) in enumerate(self._regions):
            for which, bx in (("start", self._t_to_x(r0)), ("end", self._t_to_x(r1))):
                d = abs(px - bx)
                if d < best_dist:
                    best_dist, best_idx, best_which = d, i, which

        if best_idx < 0:
            return

        if best_which == "start":
            t_min = self._regions[best_idx - 1][1] + EPSILON if best_idx > 0 else float(self._times[0])
            t_max = self._regions[best_idx][1] - EPSILON
        else:
            t_min = self._regions[best_idx][0] + EPSILON
            t_max = (self._regions[best_idx + 1][0] - EPSILON
                     if best_idx < len(self._regions) - 1 else float(self._times[-1]))

        self._drag = DragState(best_idx, best_which, t_min, t_max)

    def mouseMoveEvent(self, event):
        if self._times is None:
            return
        px = event.position().x()
        if self._drag is not None:
            t = max(self._drag.t_min, min(self._drag.t_max, self._x_to_t(px)))
            i = self._drag.region_idx
            if self._drag.which == "start":
                self._regions[i][0] = t
            else:
                self._regions[i][1] = t
            self.update()
        else:
            for r0, r1 in self._regions:
                if (abs(px - self._t_to_x(r0)) <= DRAG_THRESHOLD_PX or
                        abs(px - self._t_to_x(r1)) <= DRAG_THRESHOLD_PX):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    return
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        self._drag = None
        self.regions_changed.emit()
