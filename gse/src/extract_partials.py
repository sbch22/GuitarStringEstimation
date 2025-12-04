import os
import sys
sys.path.append(os.path.abspath(''))

from collections import Counter
import argparse
import torch

import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import json
import math
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.signal import find_peaks
from typing import Dict, List
import torchaudio
from scipy import interpolate
from utils.note_event_dataclasses import matchNote
from typing import Union


def estimate_frequency_and_amplitude(prev_phase, current_phase, k, W, H, sr, frame, window):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k

    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi

    est_freq = delta_phi / (2 * np.pi * H) * sr

    n = np.arange(W)
    dft_coeff = np.dot(frame * window, np.exp(1j * 2 * np.pi * est_freq / sr * n))
    est_amplitude = 2 * np.abs(dft_coeff) / np.sum(window)

    return est_freq, est_amplitude


def calculate_partials(
        fundamental: float,
        k_fundamental: int,
        prev_fft: np.ndarray,
        curr_fft: np.ndarray,
        fft_size: int,
        hop_size: int,
        sample_rate: float,
        beta_max: float,
        n_partials: int,
        debug: bool,
        time_frame: np.ndarray,
        window: np.ndarray,
        peak_indices: np.ndarray
) -> Dict[int, float]:
    """
    Calculate partial frequencies for a given fundamental frequency.

    Args:
        fundamental: Estimated fundamental frequency in Hz
        k_fundamental: Bin index of the fundamental frequency
        prev_fft: FFT of previous frame (for phase analysis)
        curr_fft: FFT of current frame
        fft_size: Size of the FFT
        hop_size: Hop size between frames in samples
        sample_rate: Audio sample rate in Hz
        beta_max: Maximum allowed inharmonicity coefficient
        n_partials: Maximum number of partials to calculate
        debug: Enable debug output and plotting
        time_frame: Current time domain frame
        window: Window function used for FFT
        peak_indices: Indices of spectral peaks found in current frame

    Returns:
        Dictionary mapping partial numbers to their frequencies in Hz
    """
    partials = {0: fundamental}  # Store fundamental as partial 0

    for partial_num in range(1, n_partials):
        # Calculate expected partial frequency and maximum allowed frequency considering inharmonicity
        harmonic_order = partial_num + 1
        expected_frequency = harmonic_order * fundamental
        max_inharmonic_frequency = expected_frequency * math.sqrt(1 + beta_max * harmonic_order ** 2)

        # Convert frequency to bin index
        k_partial = int(harmonic_order * k_fundamental)
        if k_partial >= fft_size // 2:  # Ensure bin is within valid range
            break

        best_partial = None
        best_amplitude = -float('inf')
        best_bin_offset = None

        # Find peak candidates within the expected frequency range
        peak_candidates = [
            p for p in peak_indices
            if expected_frequency <= p * sample_rate / fft_size <= max_inharmonic_frequency
        ]

        if peak_candidates:
            # Select peak with highest amplitude as candidate
            peak_bin = max(peak_candidates, key=lambda p: np.abs(curr_fft[p]))

            # Search in vicinity of peak bin (-2 to +8 bins)
            for bin_offset in range(-2, 9):
                test_bin = peak_bin + bin_offset
                if test_bin < 0 or test_bin >= fft_size // 2:
                    continue  # Skip invalid bins

                # Estimate precise frequency and amplitude using phase information
                partial_freq, partial_amp = estimate_frequency_and_amplitude(
                    np.angle(prev_fft[test_bin]),
                    np.angle(curr_fft[test_bin]),
                    test_bin,
                    fft_size,
                    hop_size,
                    sample_rate,
                    time_frame,
                    window
                )

                partial_amp_db = 20 * np.log10(np.abs(partial_amp))

                # Update best candidate if amplitude is sufficient
                if partial_amp_db > -40 and partial_amp_db > best_amplitude:
                    best_partial = partial_freq
                    best_amplitude = partial_amp_db
                    best_bin_offset = test_bin - peak_bin

        # Store valid partials
        if best_partial is not None:
            partials[partial_num] = best_partial
            if debug:
                print(f"Bin-Offset {best_bin_offset} used for partial {partial_num} "
                      f"with amplitude {best_amplitude:.2f} dB")

    if debug:
        print(f"Number of partials found: {len(partials)}")

    return partials



def calculate_fundamental(
        k_guess: int,
        prev_fft: np.ndarray,
        curr_fft: np.ndarray,
        fft_size: int,
        hop_size: int,
        sample_rate: float,
        time_frame: np.ndarray,
        window: np.ndarray
) -> float:
    """
    Refine fundamental frequency estimate using phase information.

    Args:
        k_guess: Initial guess for fundamental frequency bin index
        prev_fft: FFT of previous frame
        curr_fft: FFT of current frame
        fft_size: Size of the FFT
        hop_size: Hop size between frames in samples
        sample_rate: Audio sample rate in Hz
        time_frame: Current time domain frame
        window: Window function used for FFT

    Returns:
        Refined fundamental frequency estimate in Hz
    """
    best_fundamental = None
    best_amplitude = -float('inf')

    # Search vicinity of initial guess (±2 bins)
    for bin_offset in range(-2, 3):
        test_bin = k_guess + bin_offset
        if test_bin < 0 or test_bin >= fft_size // 2:
            continue  # Skip invalid bins

        # Estimate precise frequency and amplitude
        fundamental_freq, fundamental_amp = estimate_frequency_and_amplitude(
            np.angle(prev_fft[test_bin]),
            np.angle(curr_fft[test_bin]),
            test_bin,
            fft_size,
            hop_size,
            sample_rate,
            time_frame,
            window
        )

        fundamental_amp_db = 20 * np.log10(np.abs(fundamental_amp))

        # Select candidate with highest amplitude
        if fundamental_amp_db > best_amplitude:
            best_fundamental = fundamental_freq
            best_amplitude = fundamental_amp_db

    return best_fundamental

def