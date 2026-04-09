"""
Feature extraction functions for guitar string analysis.

Computes spectral centroid, relative partial amplitude deviations,
relative frequency deviations, and beta filtering/statistics.
"""

from __future__ import annotations

import numpy as np
import librosa as lib
from scipy import stats

from gse.src.feature_extraction.inharmonic_partial_tracking import kde_mode


# ── Beta filtering ────────────────────────────────────────────────────────────
def filter_betas(betas, beta_max):
    """Filter outliers from beta array using the IQR method."""
    betas = np.asarray(betas, dtype=float).ravel()

    valid = ~np.isnan(betas)
    valid_betas = betas[valid]

    if len(valid_betas) == 0:
        return []
    if len(valid_betas) == 1:
        return valid_betas.tolist()

    Q1 = np.quantile(valid_betas, 0.25)
    Q3 = np.quantile(valid_betas, 0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        return valid_betas.tolist()

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_values = valid_betas[
        (valid_betas >= lower_bound) & (valid_betas <= upper_bound)
    ].tolist()

    filtered_values = [beta for beta in filtered_values if 0.0 < beta < beta_max]

    return filtered_values


# ── Spectral centroid ─────────────────────────────────────────────────────────
def spectral_centroid_feature(audio_1d, W, H, sr):
    """
    Compute summary statistics of the spectral centroid over time.

    Returns a (9,) array: median, mean, min, max, std, var, skew, kurtosis, mode.
    """
    sc = lib.feature.spectral_centroid(y=audio_1d, sr=sr, n_fft=W, hop_length=H)

    sc_measures = np.array([
        np.nanmedian(sc),
        np.nanmean(sc),
        np.nanmin(sc),
        np.nanmax(sc),
        np.nanstd(sc),
        np.nanvar(sc),
        stats.skew(sc.ravel(), nan_policy="omit"),
        stats.kurtosis(sc.ravel(), nan_policy="omit"),
        kde_mode(sc.ravel()),
    ])
    return sc_measures


# ── Relative amplitude deviations ────────────────────────────────────────────
def relative_amplitude_deviations(partials):
    """
    Compute per-partial amplitude deviation statistics relative to the
    fundamental, plus per-partial amplitude decay slopes.

    Returns (rel_amp_measures, amp_slopes).
    """
    amps = partials.amplitudes  # (T, K)
    T, K = amps.shape

    valid = np.isfinite(amps)
    has_valid = valid.any(axis=0)

    first_idx = np.argmax(valid, axis=0)
    last_idx = (T - 1) - np.argmax(valid[::-1, :], axis=0)

    k_idx = np.arange(K)
    amps_start = np.where(has_valid, amps[first_idx, k_idx], np.nan)
    amps_end = np.where(has_valid, amps[last_idx, k_idx], np.nan)

    amp_slopes = -(amps_start - amps_end)

    rel_amps = amps - amps[:, 0:1]
    rel_amps = rel_amps[:, 1:]  # drop fundamental (always 0)

    rel_amp_measures = np.array([
        np.nanmedian(rel_amps, axis=0),
        np.nanmean(rel_amps, axis=0),
        np.nanmin(rel_amps, axis=0),
        np.nanmax(rel_amps, axis=0),
        np.nanstd(rel_amps, axis=0),
        np.nanvar(rel_amps, axis=0),
        stats.skew(rel_amps, axis=0, nan_policy="omit"),
        stats.kurtosis(rel_amps, axis=0, nan_policy="omit"),
        np.apply_along_axis(kde_mode, axis=0, arr=rel_amps),
    ])

    return rel_amp_measures, amp_slopes


# ── Relative frequency deviations ────────────────────────────────────────────
def relative_freq_deviations(partials, beta):
    """
    Compute per-partial frequency deviation from the inharmonicity model,
    relative to the measured frequency.

    Returns a (9, K-1) array of summary statistics per partial.
    """
    freqs = partials.frequencies  # (T, K)
    T, K = freqs.shape
    k = np.arange(1, K + 1)

    f0 = freqs[:, [0]]  # (T, 1)

    k = k[1:]            # (K-1,)
    freqs_k = freqs[:, 1:]

    betas = np.full((T, 1), beta)

    ideal_f_k = k * f0 * np.sqrt(1 + betas * k ** 2)

    abs_freq_deviations = ideal_f_k - freqs_k
    rel_freq_deviations = abs_freq_deviations / freqs_k

    freq_deviation_measures = np.array([
        np.nanmedian(rel_freq_deviations, axis=0),
        np.nanmean(rel_freq_deviations, axis=0),
        np.nanmin(rel_freq_deviations, axis=0),
        np.nanmax(rel_freq_deviations, axis=0),
        np.nanstd(rel_freq_deviations, axis=0),
        np.nanvar(rel_freq_deviations, axis=0),
        stats.skew(rel_freq_deviations, axis=0, nan_policy="omit"),
        stats.kurtosis(rel_freq_deviations, axis=0, nan_policy="omit"),
        np.apply_along_axis(kde_mode, axis=0, arr=rel_freq_deviations),
    ])

    return freq_deviation_measures
