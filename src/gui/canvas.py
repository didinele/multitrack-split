from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Signal

DRAG_THRESHOLD_PX = 8
EPSILON = 1.0


@dataclass
class RegionArtists:
    idx: int
    start_line: Line2D
    end_line: Line2D
    span_patch: object


@dataclass
class DragState:
    region_idx: int
    which: str  # "start" or "end"
    x_min: float
    x_max: float


class RegionEditorCanvas(FigureCanvasQTAgg):
    regions_changed = Signal()

    def __init__(self):
        fig = Figure(figsize=(15, 6), tight_layout=True)
        super().__init__(fig)
        self._ax = fig.add_subplot(111)

        self._times: Optional[np.ndarray] = None
        self._regions: list[list[float]] = []
        self._region_artists: list[RegionArtists] = []
        self._drag: Optional[DragState] = None
        self._edit_mode = False
        self._cid_press: Optional[int] = None
        self._cid_motion: Optional[int] = None
        self._cid_release: Optional[int] = None

    def update_visualization(
        self,
        times: np.ndarray,
        combined: np.ndarray,
        smoothed: np.ndarray,
        state: np.ndarray,
        raw_regions: list,
        extended_regions: list,
        ascension_spans: list,
        regions: list,
    ):
        self._times = times
        self._regions = [list(r) for r in regions]
        self._region_artists = []

        self._ax.clear()
        self._draw_static_layers(combined, smoothed, state, raw_regions, extended_regions, ascension_spans)
        self._draw_region_artists()
        self.draw_idle()

    def _draw_static_layers(
        self,
        combined: np.ndarray,
        smoothed: np.ndarray,
        state: np.ndarray,
        raw_regions: list,
        extended_regions: list,
        ascension_spans: list,
    ):
        ax = self._ax
        times = self._times

        ax.plot(times, combined, alpha=0.25, color="gray", label="Combined Features")
        ax.plot(times, smoothed, color="blue", linewidth=1.5, label="Smoothed Features")

        state_curve = state.astype(float) * np.max(smoothed) * 1.05
        ax.plot(times, state_curve, color="green", alpha=0.65, label="Hysteresis State")

        for i, (raw_start, raw_end) in enumerate(raw_regions):
            ax.axvline(raw_start, color="orange", linestyle=":", linewidth=1,
                       label="Raw Region" if i == 0 else "")
            ax.axvline(raw_end, color="orange", linestyle=":", linewidth=1)

        for i, (old_start, new_start) in enumerate(ascension_spans):
            ax.axvspan(old_start, new_start, color="yellow", alpha=0.25,
                       label="Ascension Extension" if i == 0 else "")
            ax.axvline(old_start, color="gold", linestyle="-.", linewidth=1)
            ax.axvline(new_start, color="gold", linestyle="-", linewidth=1)

        for i, (ext_start, ext_end) in enumerate(extended_regions):
            ax.axvspan(ext_start, ext_end, color="red", alpha=0.05,
                       label="Pre-pad Region" if i == 0 else "")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_title('Audio Segmentation — enable "Edit Regions" mode, then drag red boundaries')
        ax.legend(loc="upper left", fontsize="small")

    def _draw_region_artists(self):
        ax = self._ax
        for i, (start, end) in enumerate(self._regions):
            span = ax.axvspan(start, end, color="red", alpha=0.15,
                              label="Final Region" if i == 0 else "")
            s_line = ax.axvline(start, color="red", linestyle="--", linewidth=1.5)
            e_line = ax.axvline(end, color="red", linestyle="--", linewidth=1.5)
            self._region_artists.append(RegionArtists(
                idx=i, start_line=s_line, end_line=e_line, span_patch=span,
            ))

    def set_edit_mode(self, enabled: bool, toolbar: NavigationToolbar2QT):
        self._edit_mode = enabled
        if enabled:
            mode = toolbar.mode
            if hasattr(mode, "name"):
                if mode.name == "PAN":
                    toolbar.pan()
                elif mode.name == "ZOOM":
                    toolbar.zoom()
            self._cid_press = self.mpl_connect("button_press_event", self._on_press)
            self._cid_motion = self.mpl_connect("motion_notify_event", self._on_motion)
            self._cid_release = self.mpl_connect("button_release_event", self._on_release)
        else:
            for cid in (self._cid_press, self._cid_motion, self._cid_release):
                if cid is not None:
                    self.mpl_disconnect(cid)
            self._cid_press = self._cid_motion = self._cid_release = None
            self._drag = None

    def _hit_threshold_in_data(self) -> float:
        inv = self._ax.transData.inverted()
        p0 = inv.transform((0, 0))
        p1 = inv.transform((DRAG_THRESHOLD_PX, 0))
        return abs(p1[0] - p0[0])

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.button != 1 or event.xdata is None:
            return
        threshold = self._hit_threshold_in_data()
        max_time = float(self._times[-1]) if self._times is not None else 0.0

        for i, _ in enumerate(self._region_artists):
            start, end = self._regions[i]

            if abs(event.xdata - start) < threshold:
                prev_end = self._regions[i - 1][1] if i > 0 else 0.0
                self._drag = DragState(i, "start", prev_end + EPSILON, end - EPSILON)
                return

            if abs(event.xdata - end) < threshold:
                next_start = self._regions[i + 1][0] if i < len(self._regions) - 1 else max_time
                self._drag = DragState(i, "end", start + EPSILON, next_start - EPSILON)
                return

    def _on_motion(self, event):
        if self._drag is None or event.xdata is None:
            return
        self._apply_drag(event.xdata)

    def _on_release(self, event):
        if self._drag is None:
            return
        if event.xdata is not None:
            self._apply_drag(event.xdata)
        self._drag = None
        self.regions_changed.emit()

    def _apply_drag(self, raw_x: float):
        d = self._drag
        x = max(d.x_min, min(d.x_max, raw_x))
        i = d.region_idx
        ra = self._region_artists[i]

        if d.which == "start":
            self._regions[i][0] = x
            ra.start_line.set_xdata([x, x])
            xy = ra.span_patch.get_xy()
            xy[0, 0] = xy[1, 0] = xy[4, 0] = x
            ra.span_patch.set_xy(xy)
        else:
            self._regions[i][1] = x
            ra.end_line.set_xdata([x, x])
            xy = ra.span_patch.get_xy()
            xy[2, 0] = xy[3, 0] = x
            ra.span_patch.set_xy(xy)

        self.draw_idle()

    def get_regions(self) -> list[tuple[float, float]]:
        return [(r[0], r[1]) for r in self._regions]
