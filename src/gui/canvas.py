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
_ZOOM_FACTOR = 1.25
_MIN_VIEW_SPAN = 1.0  # minimum 1 second when zoomed in

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

        # View state — None means full view
        self._view_t0: Optional[float] = None
        self._view_t1: Optional[float] = None

        # Caches — invalidated on resize, data change, or view change
        self._bucket_cache: Optional[np.ndarray] = None
        self._bucket_cache_key: tuple = (0, 0.0, 0.0)  # (w, view_t0, view_t1)
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

    def _get_view_bounds(self) -> tuple[float, float]:
        if self._times is None:
            return (0.0, 1.0)
        t_full_0, t_full_1 = float(self._times[0]), float(self._times[-1])
        t0 = self._view_t0 if self._view_t0 is not None else t_full_0
        t1 = self._view_t1 if self._view_t1 is not None else t_full_1
        return (t0, t1)

    def _t_to_x(self, t: float) -> float:
        t0, t1 = self._get_view_bounds()
        return (t - t0) / (t1 - t0) * self.width()

    def _x_to_t(self, x: float) -> float:
        t0, t1 = self._get_view_bounds()
        return t0 + (x / self.width()) * (t1 - t0)

    # ------------------------------------------------------------------
    # Waveform caches
    # ------------------------------------------------------------------

    def _invalidate_waveform_cache(self):
        self._bucket_cache = None
        self._bucket_cache_key = (0, 0.0, 0.0)
        self._active_cache = None
        self._active_cache_sz = (0, 0)

    def _get_buckets(self, w: int) -> np.ndarray:
        vt0, vt1 = self._get_view_bounds()
        key = (w, vt0, vt1)
        if self._bucket_cache is not None and self._bucket_cache_key == key:
            return self._bucket_cache

        data = self._combined
        if data is None or len(data) == 0:
            result = np.zeros(w)
        else:
            # Slice to the visible time range for accurate per-pixel sampling
            i_start = max(0, int(np.searchsorted(self._times, vt0, side='left')))
            i_end = min(len(data), int(np.searchsorted(self._times, vt1, side='right')))
            data_slice = data[i_start:i_end]

            if len(data_slice) == 0:
                result = np.zeros(w)
            else:
                n = len(data_slice)
                edges = np.linspace(0, n, w + 1).astype(int)
                starts = edges[:-1]
                unique = np.concatenate(([True], starts[1:] > starts[:-1]))
                if unique.all():
                    result = np.maximum.reduceat(data_slice, starts).astype(float)
                else:
                    result = np.interp(np.linspace(0, n - 1, w), np.arange(n), data_slice)
                mx = result.max()
                if mx > 0:
                    result /= mx

        self._bucket_cache = result
        self._bucket_cache_key = key
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
    # Zoom / pan
    # ------------------------------------------------------------------

    def _zoom_by(self, factor: float, pivot_x: float):
        """Zoom by factor (>1 = zoom in) centered on the given pixel x."""
        if self._times is None:
            return
        t_full_0 = float(self._times[0])
        t_full_1 = float(self._times[-1])
        full_span = t_full_1 - t_full_0

        vt0, vt1 = self._get_view_bounds()
        span = vt1 - vt0
        t_pivot = vt0 + (pivot_x / max(1, self.width())) * span

        new_span = max(_MIN_VIEW_SPAN, span / factor)

        # If zooming out past the full range, reset to full view
        if new_span >= full_span:
            self._view_t0 = None
            self._view_t1 = None
            self._invalidate_waveform_cache()
            self.update()
            return

        # Keep the pivot point fixed under the cursor
        frac = (pivot_x / max(1, self.width()))
        new_t0 = t_pivot - frac * new_span
        new_t1 = new_t0 + new_span

        # Clamp to full range
        if new_t0 < t_full_0:
            new_t0 = t_full_0
            new_t1 = new_t0 + new_span
        if new_t1 > t_full_1:
            new_t1 = t_full_1
            new_t0 = new_t1 - new_span

        self._view_t0 = max(t_full_0, new_t0)
        self._view_t1 = min(t_full_1, new_t1)
        self._invalidate_waveform_cache()
        self.update()

    def _pan_by(self, delta_px: float):
        """Pan the view by delta_px pixels (positive = forward in time)."""
        if self._times is None or (self._view_t0 is None and self._view_t1 is None):
            return
        t_full_0 = float(self._times[0])
        t_full_1 = float(self._times[-1])
        vt0, vt1 = self._get_view_bounds()
        span = vt1 - vt0
        delta_t = (delta_px / max(1, self.width())) * span

        new_t0 = vt0 + delta_t
        new_t1 = vt1 + delta_t

        if new_t0 < t_full_0:
            new_t0 = t_full_0
            new_t1 = new_t0 + span
        if new_t1 > t_full_1:
            new_t1 = t_full_1
            new_t0 = new_t1 - span

        self._view_t0 = new_t0
        self._view_t1 = new_t1
        self._invalidate_waveform_cache()
        self.update()

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

        # Region boundary handles — only draw if within the current view
        pen = QPen(_HANDLE)
        pen.setWidth(2)
        painter.setPen(pen)
        for r0, r1 in self._regions:
            x0 = round(self._t_to_x(r0))
            x1 = round(self._t_to_x(r1))
            if 0 <= x0 <= w:
                painter.drawLine(x0, _PAD_TOP, x0, _PAD_TOP + wave_h)
            if 0 <= x1 <= w:
                painter.drawLine(x1, _PAD_TOP, x1, _PAD_TOP + wave_h)

        self._draw_time_axis(painter, w, h, _PAD_TOP + wave_h)
        painter.end()

    def _draw_time_axis(self, painter: QPainter, w: int, h: int, y_base: int):
        t0, t1 = self._get_view_bounds()
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
    # Mouse events — drag handles, zoom reset
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

    def mouseDoubleClickEvent(self, event):
        if self._times is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._view_t0 = None
        self._view_t1 = None
        self._invalidate_waveform_cache()
        self.update()

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

    def wheelEvent(self, event):
        if self._times is None:
            return
        delta = event.angleDelta().y()
        if not delta:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = _ZOOM_FACTOR if delta > 0 else 1.0 / _ZOOM_FACTOR
            self._zoom_by(factor, event.position().x())
        else:
            # Each scroll notch (120 units) pans 20% of the visible width
            pan_px = -(delta / 120.0) * (self.width() * 0.2)
            self._pan_by(pan_px)
        event.accept()
