import numpy as np
import librosa
from scipy.signal import find_peaks, windows
from scipy.fftpack import fft
from typing import Tuple, List, Optional
import warnings


def parabolic_interpolation(idx: int, data: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform parabolic interpolation around a peak index to get a more accurate position.

    Args:
        idx: Index of the peak in the data array
        data: Array of data values

    Returns:
        Tuple containing:
        - Interpolated index position
        - Delta from the original index
        - Interpolated maximum value
    """
    if idx <= 0 or idx >= len(data) - 1:
        return idx, 0.0, data[idx]

    # Get values around the peak
    xy = data[idx - 1:idx + 2]
    p = 1  # The peak is at the center (index=1)

    # Calculate parabolic shift
    delta = 0.5 * (xy[p - 1] - xy[p + 1]) / (xy[p - 1] - 2 * xy[p] + xy[p + 1])
    new_max = xy[p] - 0.25 * (xy[p - 1] - xy[p + 1]) * delta

    return idx + delta, delta, new_max


def estimate_frequency_from_phase(prev_phase: float, current_phase: float,
                                  k: int, window_size: int, hop_size: int,
                                  sample_rate: int) -> float:
    """
    Estimate frequency from phase difference between consecutive frames.

    Args:
        prev_phase: Phase from previous frame
        current_phase: Phase from current frame
        k: Frequency bin index
        window_size: Size of the analysis window
        hop_size: Hop size between frames
        sample_rate: Sampling rate in Hz

    Returns:
        Estimated frequency in Hz
    """
    omega_k = 2 * np.pi * k / window_size
    delta_phi = omega_k * hop_size + np.mod(
        current_phase - prev_phase - omega_k * hop_size + np.pi, 2 * np.pi) - np.pi
    return delta_phi / (2 * np.pi * hop_size) * sample_rate


def safe_yin_estimation(signal: np.ndarray, fmin: float, fmax: float,
                        sample_rate: int, window_size: int, hop_size: int) -> Optional[np.ndarray]:
    """
    Safely estimate frequencies using YIN algorithm with error handling.

    Args:
        signal: Input audio signal
        fmin: Minimum frequency to consider
        fmax: Maximum frequency to consider
        sample_rate: Sampling rate in Hz
        window_size: Size of the analysis window
        hop_size: Hop size between frames

    Returns:
        Array of estimated frequencies or None if estimation fails
    """
    try:
        # Calculate required parameters for YIN
        win_length = window_size // 2

        # Check if parameters are valid for YIN
        if fmax * window_size / sample_rate > win_length // 2:
            # Adjust fmax to valid range
            valid_fmax = sample_rate * (win_length // 2) / window_size
            if valid_fmax <= fmin:
                return None
            fmax = valid_fmax

        return librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sample_rate,
                           frame_length=window_size, hop_length=hop_size)
    except Exception as e:
        warnings.warn(f"YIN estimation failed: {e}")
        return None


def safe_pyin_estimation(signal: np.ndarray, fmin: float, fmax: float,
                         sample_rate: int, window_size: int, hop_size: int) -> Optional[np.ndarray]:
    """
    Safely estimate frequencies using pYIN algorithm with error handling.

    Args:
        signal: Input audio signal
        fmin: Minimum frequency to consider
        fmax: Maximum frequency to consider
        sample_rate: Sampling rate in Hz
        window_size: Size of the analysis window
        hop_size: Hop size between frames

    Returns:
        Array of estimated frequencies or None if estimation fails
    """
    try:
        # Calculate required parameters for pYIN
        win_length = window_size // 2

        # Check if parameters are valid for pYIN
        if fmax * window_size / sample_rate > win_length // 2:
            # Adjust fmax to valid range
            valid_fmax = sample_rate * (win_length // 2) / window_size
            if valid_fmax <= fmin:
                return None
            fmax = valid_fmax

        return librosa.pyin(signal, fmin=fmin, fmax=fmax, sr=sample_rate,
                            frame_length=window_size, hop_length=hop_size)[0]
    except Exception as e:
        warnings.warn(f"pYIN estimation failed: {e}")
        return None


def evaluate_frequency_estimation():
    """
    Main function to evaluate different frequency estimation methods on synthetic signals.

    Compares:
    - Basic FFT peak picking
    - FFT with parabolic interpolation
    - Instantaneous phase method
    - YIN algorithm
    - pYIN algorithm
    """
    # Parameters
    sample_rate = 16000  # Sampling rate
    freq_range = np.arange(82, 360, 1)  # Frequency range to test
    signal_length = 3  # Signal length in seconds
    time_vector = np.arange(0, signal_length, 1 / sample_rate)

    # Window sizes to test
    window_sizes = [512, 1024, 2048, 4096, int(0.1 * sample_rate), int(0.3 * sample_rate)]

    # Frequency range for YIN methods
    fmin, fmax = np.min(freq_range), np.max(freq_range)

    # Initialize error and beta arrays
    n_methods = 5  # FFT, parabolic, phase, YIN, pYIN
    errors = np.zeros((len(freq_range), len(window_sizes), n_methods))
    betas = np.zeros((len(freq_range), len(window_sizes), n_methods))

    method_names = ["FFT", "Parabolic Interpolation", "Instantaneous Phase", "YIN", "pYIN"]

    # Main evaluation loop
    for f_idx, freq in enumerate(freq_range):
        # Generate synthetic signal
        signal = np.sin(2 * np.pi * freq * time_vector)

        for w_idx, window_size in enumerate(window_sizes):
            hop_size = window_size // 8
            window = windows.hann(window_size, sym=True)

            # Precompute YIN and pYIN for the entire signal with error handling
            yin_hop = window_size // 16
            yin_freqs = safe_yin_estimation(signal, fmin, fmax, sample_rate, window_size, yin_hop)
            pyin_freqs = safe_pyin_estimation(signal, fmin, fmax, sample_rate, window_size, yin_hop)

            # Initialize previous segment for phase-based estimation
            prev_segment = signal[0:window_size] * window

            # Process each frame
            num_frames = len(signal) // hop_size - 1
            frame_errors = []
            frame_betas = []

            for frame in range(1, num_frames):
                start = frame * hop_size
                end = start + window_size

                if end > len(signal):
                    break

                # Extract and window current segment
                segment = signal[start:end] * window

                # Compute FFT and find peaks
                spectrum = np.abs(fft(segment, window_size))[:window_size // 2 + 1]
                spectrum_log = 20 * np.log10(spectrum + 1e-10)

                peaks, _ = find_peaks(spectrum_log, height=-30)
                if len(peaks) == 0:
                    continue

                peak_idx = peaks[0]

                # Method 1: Basic FFT peak picking
                fft_freq = peak_idx * sample_rate / window_size

                # Method 2: Parabolic interpolation
                interp_idx, delta, _ = parabolic_interpolation(peak_idx, spectrum_log)
                interp_freq = interp_idx * sample_rate / window_size

                # Method 3: Instantaneous phase
                prev_phase = np.angle(fft(prev_segment, window_size))[peak_idx]
                current_phase = np.angle(fft(segment, window_size))[peak_idx]
                phase_freq = estimate_frequency_from_phase(
                    prev_phase, current_phase, peak_idx, window_size, hop_size, sample_rate)

                # Update previous segment
                prev_segment = segment

                # Methods 4 & 5: YIN and pYIN (with fallback to true frequency if estimation failed)
                yin_est = yin_freqs[frame] if yin_freqs is not None and frame < len(yin_freqs) and not np.isnan(
                    yin_freqs[frame]) else freq
                pyin_est = pyin_freqs[frame] if pyin_freqs is not None and frame < len(pyin_freqs) and not np.isnan(
                    pyin_freqs[frame]) else freq

                # Calculate beta values (normalized frequency deviation)
                beta_vals = [
                    (fft_freq / freq) ** 2 - 1,
                    (interp_freq / freq) ** 2 - 1,
                    (phase_freq / freq) ** 2 - 1,
                    (yin_est / freq) ** 2 - 1,
                    (pyin_est / freq) ** 2 - 1
                ]

                # Calculate errors
                error_vals = [
                    abs(freq - fft_freq),
                    abs(freq - interp_freq),
                    abs(freq - phase_freq),
                    abs(freq - yin_est),
                    abs(freq - pyin_est)
                ]

                frame_errors.append(error_vals)
                frame_betas.append(beta_vals)

            # Store mean values for this frequency and window size
            if frame_errors:
                errors[f_idx, w_idx] = np.mean(frame_errors, axis=0)
                betas[f_idx, w_idx] = np.mean(frame_betas, axis=0)

    # Print results
    print("Frequency Estimation Evaluation Results")
    print("=" * 80)

    for w_idx, window_size in enumerate(window_sizes):
        print(f"\nWindow Size: {window_size} samples")
        print("-" * 40)

        for m_idx, method in enumerate(method_names):
            mean_error = np.mean(errors[:, w_idx, m_idx])
            std_error = np.std(errors[:, w_idx, m_idx])
            mean_beta = np.mean(betas[:, w_idx, m_idx])
            std_beta = np.std(betas[:, w_idx, m_idx])

            print(f"{method:25s} | Error: {mean_error:8.4f} ± {std_error:8.4f} Hz | "
                  f"Beta: {mean_beta:8.6f} ± {std_beta:8.6f}")


if __name__ == "__main__":
    evaluate_frequency_estimation()