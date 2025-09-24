import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Add project/amt/src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../amt/src")))

from amt.src.betaDistributions import (
    estimate_inharmonicity_coefficients,
    flatten_recursive,
    noteToFreq
)

from amt.src.utils.note_event_dataclasses import matchNote


def make_synthetic_matchnote(onset_s: float, offset_s: float, midi_pitch: int, sr: int) -> matchNote:
    freq = noteToFreq(midi_pitch)
    contour = [(onset_s, freq), (offset_s, freq)]
    return matchNote(
        is_drum=False,
        program=0,
        onset=onset_s,
        onsetDiff=0.0,
        offset=offset_s,
        pitch=midi_pitch,
        velocity=100,
        contour=contour,
        string_index=0
    )


def generate_signal(fundamental: float,
                    n_partials: int,
                    sr: int,
                    beta: float,
                    length: int,
                    pitch_lift: float = 0.0) -> np.ndarray:
    """
    Generate a synthetic harmonic signal with optional inharmonicity and pitch lift.

    Args:
        fundamental: Fundamental frequency in Hz.
        n_partials: Number of partials (harmonics) to include.
        sr: Sampling rate in Hz.
        beta: Inharmonicity coefficient.
        length: Length of the signal in samples.
        pitch_lift: Relative frequency increase over time (e.g., 0.05 = +5% at end).

    Returns:
        Synthesized signal as a NumPy array of shape (length,).
    """
    t = np.arange(length) / sr  # Time axis in seconds
    signal = np.zeros(length)

    # Linear frequency factor (1.0 → 1.05 for +5% pitch lift)
    freq_factor = 1.0 + (pitch_lift * t / t[-1])

    # Fundamental
    signal += np.sin(2 * np.pi * fundamental * freq_factor * t)

    # Overtones (starting from the 2nd harmonic)
    for i in range(2, n_partials + 1):
        # Inharmonic partial frequency at t=0
        base_partial_freq = i * fundamental * math.sqrt(1 + beta * (i ** 2))

        # Apply pitch lift scaling
        partial_freq = base_partial_freq * freq_factor

        # Amplitude roll-off (example: 1/sqrt(i))
        amplitude = 1.0 / (i ** 0.5)

        # Add partial
        signal += amplitude * np.sin(2 * np.pi * partial_freq * t)

    return signal

# Main
def main():
    length = 88200
    n_partials = 25
    sr = 44100

    midi_pitch = 69  # A4
    onset_s = 0.0
    offset_s = length / sr

    notes = [make_synthetic_matchnote(onset_s, offset_s, midi_pitch, sr)]

    # Signal possible to integrate a pitch-lift to simulate pitch-shifts in real signal
    sig = generate_signal(440, n_partials, 44100, 1e-4, 88200, pitch_lift=0.05)
    sig /= np.max(abs(sig))


    estimated_betas = estimate_inharmonicity_coefficients(sig, sr, notes, debug=False)
    betas = flatten_recursive(estimated_betas)

    # Statistical analysis
    mean_beta = np.mean(betas)
    median_beta = np.median(betas)
    std_beta = np.std(betas, ddof=1)
    var_beta = np.var(betas, ddof=1)

    # KDE Calculation
    beta_values = np.array([b for b in betas if 0 <= b <= 0.001])  # Filter values
    x_grid = np.linspace(0, 0.001, 1000)  # Create evaluation grid

    if len(beta_values) > 1:  # Ensure we have enough data points
        kde = gaussian_kde(beta_values)
        kde_values = kde(x_grid)

        # Normalize KDE to area=1
        area = np.trapz(kde_values, x_grid)
        kde_values /= area

        # Find mode (de-max)
        mode_beta = x_grid[np.argmax(kde_values)]
    else:
        mode_beta = None

    print("=== Beta Statistics ===")
    print(f"Number of values: {len(betas)}")
    print(f"Estimated Beta (Mean): {mean_beta:.6e}")
    print(f"Estimated Beta (Median): {median_beta:.6e}")
    if mode_beta is not None:
        print(f"Estimated Beta (Mode): {mode_beta:.6e}")
    print(f"Standard Deviation: {std_beta:.6e}")
    print(f"Variance: {var_beta:.6e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(betas, bins=20, density=True, color="steelblue",
             edgecolor="black", alpha=0.7, label="Histogram")

    if mode_beta is not None:
        plt.plot(x_grid, kde_values, color='darkorange', linewidth=2, label="KDE")
        plt.axvline(mode_beta, color='green', linestyle='--',
                    label=f'Mode = {mode_beta:.2e}')

    plt.axvline(median_beta, color='red', linestyle='--',
                label=f'Median = {median_beta:.2e}')
    plt.axvline(mean_beta, color='purple', linestyle='--',
                label=f'Mean = {mean_beta:.2e}')

    plt.xlabel(r"$\beta$")
    plt.ylabel("Density")
    plt.title("Beta Distribution with Statistical Measures")
    plt.legend()
    plt.xlim(0, 0.0002)
    plt.grid(True, alpha=0.3)
    plt.show()



# %%
if __name__ == "__main__":
    main()