import matplotlib.pyplot as plt
import numpy as np

def plot_debug(times, combined, smoothed, state, raw_regions, final_regions, ascension_spans, output_path: str):
    plt.figure(figsize=(15, 6))

    plt.plot(times, combined, alpha=0.25, label='Combined Features', color='gray')
    plt.plot(times, smoothed, label='Smoothed Features', color='blue', linewidth=1.5)

    # Overlay the binary state slightly scaled so the active/inactive state is visible on the same axis
    state_curve = state.astype(float) * np.max(smoothed) * 1.05
    plt.plot(times, state_curve, label='Hysteresis State', color='green', alpha=0.65)

    for i, (start, end) in enumerate(final_regions):
        plt.axvspan(start, end, color='red', alpha=0.15, label='Final Region' if i == 0 else "")
        plt.axvline(start, color='red', linestyle='--', linewidth=1)
        plt.axvline(end, color='red', linestyle='--', linewidth=1)

    for i, (raw_start, raw_end) in enumerate(raw_regions):
        plt.axvline(raw_start, color='orange', linestyle=':', linewidth=1, label='Raw Region Start' if i == 0 else "")
        plt.axvline(raw_end, color='orange', linestyle=':', linewidth=1, label='Raw Region End' if i == 0 else "")

    for i, (old_start, new_start) in enumerate(ascension_spans):
        plt.axvspan(old_start, new_start, color='yellow', alpha=0.25, label='Ascension Extension' if i == 0 else "")
        plt.axvline(old_start, color='gold', linestyle='-.', linewidth=1)
        plt.axvline(new_start, color='gold', linestyle='-', linewidth=1)

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Audio Segmentation Debug Plot')
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
