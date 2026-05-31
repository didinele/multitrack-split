import numpy as np
from scipy.ndimage import uniform_filter1d

def smooth_features(features: np.ndarray, window_size_frames: int):
    return uniform_filter1d(features, size=window_size_frames)

def apply_hysteresis(features: np.ndarray, start_thresh: float, stop_thresh: float) -> np.ndarray:
    """
    Apply dual-threshold hysteresis.
    Returns a binary array of active (1) or inactive (0) states.
    """
    active = False
    state = np.zeros_like(features, dtype=bool)
    
    for i, val in enumerate(features):
        if val > start_thresh:
            active = True
        elif val < stop_thresh:
            active = False
        state[i] = active
        
    return state

def get_regions(state: np.ndarray, times: np.ndarray, min_active_sec: float, min_silence_sec: float):
    """
    Convert a boolean state array into (start, end) time tuples.
    Merges nearby regions and drops excessively short ones.
    """
    changes = np.diff(state.astype(int))
    starts = np.where(changes == 1)[0] + 1
    stops = np.where(changes == -1)[0] + 1
    
    if state[0]:
        starts = np.insert(starts, 0, 0)
    if state[-1]:
        stops = np.append(stops, len(state) - 1)
        
    regions = []
    for start_idx, stop_idx in zip(starts, stops):
        start_time = times[start_idx]
        stop_time = times[stop_idx]
        regions.append([start_time, stop_time])
        
    if not regions:
        return []
        
    # Merge regions separated by silence less than min_silence_sec
    merged = [regions[0]]
    for r in regions[1:]:
        prev = merged[-1]
        if r[0] - prev[1] < min_silence_sec:
            prev[1] = r[1]
        else:
            merged.append(r)
            
    # Filter out regions shorter than min_active_sec
    final_regions = [r for r in merged if r[1] - r[0] >= min_active_sec]
    
    return final_regions

# an attempt at avoiding magic numbers. should these perhaps be configurable via CLI?
MIN_RISE = 0.005
MAX_DOWN_FRAC = 0.4
MAX_NEGATIVE_STEP = 0.04

def extend_regions_for_ascending_start(regions, times: np.ndarray, smoothed: np.ndarray, max_lookback_sec: float):
    """Extend region starts when the preceding signal is mostly ascending.

    This function scans backwards from each detected start and looks for a prior point where the smoothed energy signal
    begins rising steadily toward the raw start. The heuristic is intentionally tolerant of small dips, but it rejects
    potential extensions that contain large downward drops or too many negative steps.
    """
    if not regions or len(times) < 2:
        return regions, []

    sample_interval = float(np.median(np.diff(times)))
    lookback_frames = int(max_lookback_sec / sample_interval)
    extended = []
    ascension_spans = []

    for start, end in regions:
        start_idx = np.searchsorted(times, start)
        if start_idx <= 1 or lookback_frames <= 0:
            extended.append((start, end))
            continue

        search_start = max(0, start_idx - lookback_frames)
        window = smoothed[search_start:start_idx + 1]
        if window.size <= 1:
            extended.append((start, end))
            continue

        best_candidate = start_idx
        for rel_idx in range(0, len(window) - 1):
            segment = window[rel_idx:]
            diffs = np.diff(segment)
            if diffs.size == 0:
                continue

            # Total rise from candidate to current start
            rise = float(segment[-1] - segment[0])
            # Fraction of steps that dip instead of rise
            down_frac = np.sum(diffs < -1e-4) / len(diffs)
            # Largest single negative step in the candidate window
            max_drop = np.max(np.maximum(0.0, -diffs))

            if rise < MIN_RISE:
                continue
            if down_frac > MAX_DOWN_FRAC:
                continue
            if max_drop > MAX_NEGATIVE_STEP:
                continue

            best_candidate = search_start + rel_idx
            break

        if best_candidate == start_idx:
            extended.append((start, end))
            continue

        extended_start = float(times[best_candidate])
        extended.append((extended_start, end))
        ascension_spans.append((start, extended_start))

    return extended, ascension_spans

def add_padding(regions, pre_pad: float, post_pad: float, max_time: float):
    padded = []
    for start, end in regions:
        padded_start = max(0.0, start - pre_pad)
        padded_end = min(max_time, end + post_pad)
        padded.append((padded_start, padded_end))

    return padded
