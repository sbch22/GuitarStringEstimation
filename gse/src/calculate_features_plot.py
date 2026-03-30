import os
import sys
sys.path.append(os.path.abspath(''))

import multiprocessing as mp
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
import pyfar as pf
import librosa as lib
from scipy import stats
from scipy.signal import medfilt
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from utils.FeatureNote_dataclass import FilterReason
from utils.FeatureNote_dataclass import Features
from gse.src.utils.FeatureNote_dataclass import Partials
from find_partials import note_audio_preprocess


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D




""" FEATURES """
def filter_betas(betas, beta_max):
    """Filter outliers from beta array using IQR method."""
    valid = ~np.isnan(betas)
    valid_betas = betas[valid]

    # Handle empty case
    if len(valid_betas) == 0:
        return []

    # Handle single value case
    if len(valid_betas) == 1:
        return valid_betas.tolist()

    # IQR filter
    Q1 = np.quantile(valid_betas, 0.25)
    Q3 = np.quantile(valid_betas, 0.75)
    IQR = Q3 - Q1

    # Handle zero IQR (all values identical)
    if IQR == 0:
        return valid_betas.tolist()

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_values = valid_betas[
        (valid_betas >= lower_bound) & (valid_betas <= upper_bound)
        ].tolist()

    # filter to remove invalid betas before postprocessing
    filtered_values = [beta for beta in filtered_values if 0.0 < beta < beta_max]

    return filtered_values

def _debug_search_window(
    amp_region: np.ndarray,
    freq_region: np.ndarray,
    f_k: float,
    t: int,
    p_idx: int,
    peaks_local: np.ndarray,
    fft_norm_region: np.ndarray | None = None,
) -> None:
    """
    Small diagnostic plot for one search window.
    amp_region      : raw (un-zeroed) normalised magnitude slice
    fft_norm_region : the zeroed slice actually used for peak-picking
    """
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(freq_region, amp_region, color="steelblue", label="amp (raw norm)")

    if fft_norm_region is not None:
        ax.plot(freq_region, fft_norm_region,
                color="orange", linestyle="--", label="amp (zeroed)")

    if peaks_local.size > 0:
        ax.plot(freq_region[peaks_local], amp_region[peaks_local],
                "v", color="lime", ms=8, label="peaks found")

    ax.axvline(f_k, color="red", lw=1.2, linestyle=":", label=f"f_k={f_k:.1f} Hz")
    ax.set_title(f"frame {t}, partial {p_idx+1}  —  expected {f_k:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Norm. magnitude")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def spectral_centroid_feature(audio_1d, W, H, sr):
    """audio_1d: raw 1-D note audio, shape (N_samples,)"""
    sc = lib.feature.spectral_centroid(y=audio_1d, sr=sr, n_fft=W, hop_length=H)
    # sc shape: (1, n_frames) — librosa standard output

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
    ])  # shape: (9,) — fixed, note-independent
    return sc_measures

def relative_amplitude_deviations(partials):
    amps = partials.amplitudes  # (T, K)
    T, K = amps.shape

    valid = np.isfinite(amps)         # (T, K)
    has_valid = valid.any(axis=0)     # (K,) — guard for all-NaN partials

    # First valid T index per partial: argmax stops at first True
    first_idx = np.argmax(valid, axis=0)                          # (K,)
    # Last valid T index per partial: flip T axis, then mirror back
    last_idx = (T - 1) - np.argmax(valid[::-1, :], axis=0)       # (K,)

    k_idx = np.arange(K)
    amps_start = np.where(has_valid, amps[first_idx, k_idx], np.nan)  # (K,)
    amps_end   = np.where(has_valid, amps[last_idx,  k_idx], np.nan)  # (K,)

    amp_slopes = - (amps_start - amps_end)  # (K,) — NaN where partial fully absent
    rel_amps = amps - amps[:, 0:1]  # dB-Differenz zu f0
    rel_amps = rel_amps[:, 1:]  # Partial 0 raus (immer = 0, keine Info)

    median_amps = np.nanmedian(
        rel_amps, axis=0
    )  # (K,)
    mean_amps = np.nanmean(
        rel_amps, axis=0
    )  # (K,)
    min_amps = np.nanmin(
        rel_amps, axis=0
    )  # (K,)
    max_amps = np.nanmax(
        rel_amps, axis=0
    )  # (K,)
    std_amps = np.nanstd(
        rel_amps, axis=0
    )  # (K,)
    var_amps = np.nanvar(
        rel_amps, axis=0
    )  # (K,)

    rel_amp_measures = np.array([
        median_amps,
        mean_amps,
        min_amps,
        max_amps,
        std_amps,
        var_amps,
        stats.skew(rel_amps, axis=0, nan_policy="omit"),
        stats.kurtosis(rel_amps, axis=0, nan_policy="omit"),
        np.apply_along_axis(kde_mode, axis=0, arr=rel_amps),
    ])

    return rel_amp_measures, amp_slopes

def relative_freq_deviations(partials, beta):
    freqs = partials.frequencies      # (T, K)
    T, K = freqs.shape
    k = np.arange(1, K + 1)           # (K,)

    f0 = freqs[:, [0]]                # (T, 1)

    # Remove fundamental from harmonic axis
    k = k[1:]                         # (K-1,)
    freqs_k = freqs[:, 1:]            # (T, K-1)

    betas = np.full((T, 1), beta)     # (T, 1)

    ideal_f_k = (
        k * f0 * np.sqrt(1 + betas * k**2)
    )                                 # (T, K-1)

    # Absolute deviation
    abs_freq_deviations = ideal_f_k - freqs_k   # (T, K-1)
    rel_freq_deviations = abs_freq_deviations / freqs_k


    # statistical measures
    median_freq_deviations = np.nanmedian(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)
    mean_freq_deviations = np.nanmean(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)
    min_freq_deviations = np.nanmin(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)
    max_freq_deviations = np.nanmax(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)
    std_freq_deviations = np.nanstd(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)
    var_freq_deviations = np.nanvar(
        rel_freq_deviations,
        axis=0
    )  # (K-1,)

    freq_deviation_measures = np.array([
        median_freq_deviations,
        mean_freq_deviations,
        min_freq_deviations,
        max_freq_deviations,
        std_freq_deviations,
        var_freq_deviations,
        stats.skew(rel_freq_deviations, axis=0, nan_policy="omit"),
        stats.kurtosis(rel_freq_deviations, axis=0, nan_policy="omit"),
        np.apply_along_axis(kde_mode, axis=0, arr=rel_freq_deviations)
    ])

    return freq_deviation_measures


def kde_mode(data):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    # not enough data
    if len(data) < 5:
        return np.nan

    # not enough variation
    if np.unique(data).size < 3:
        return np.nan

    # near-zero variance
    if np.std(data) < 1e-10:
        return data.mean()

    try:
        kde = stats.gaussian_kde(data, bw_method="scott")
        x = np.linspace(data.min(), data.max(), 1000)
        return x[np.argmax(kde(x))]
    except np.linalg.LinAlgError:
        return np.nan


def estimate_inharmonicity_coefficient_all_frets(partials, min_partials=4):
    freqs = partials.frequencies  # (T, K)
    T, K = freqs.shape
    betas = np.full(T, np.nan)
    k_full = np.arange(1, K + 1)

    for i in range(T):
        f_k = freqs[i]

        valid = np.isfinite(f_k)
        if np.sum(valid) < 2:
            continue

        k = k_full[valid]
        f = f_k[valid]

        # if f0 available -> otherwise estimate by common denominator
        if valid[0] and k[0] == 1:
            f0 = f[0]  # direkt bekannt

        # Eq. (15): ck
        c_k = f - k * f0

        # Eq. (16): polynomial fit
        X = np.vstack([k**3, k, np.ones_like(k)]).T

        try:
            a, b, c = np.linalg.lstsq(X, c_k, rcond=None)[0]

            # Eq. (17): beta
            beta = 2 * a / (f0 + b)

            # sanity checks
            if np.isfinite(beta) and beta > 0:
                betas[i] = beta

        except np.linalg.LinAlgError:
            continue

        # ── Plot (nur wenn ≥ 15 gültige Partiale) ────────────────────────
        if i ==20:
            k_plot = np.linspace(1, K, 400)
            poly_curve = a * k_plot**3 + b * k_plot + c

            ACCENT  = "#1565c0"
            DOT_NAN = "#c62828"
            GRID    = "#d0d0d0"

            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            # ungültige Partiale als Rug
            k_invalid = k_full[~valid]
            if len(k_invalid):
                ax.scatter(k_invalid, np.zeros(len(k_invalid)),
                           marker="|", s=140, linewidths=1.8, color=DOT_NAN,
                           alpha=0.85, zorder=3, clip_on=False,
                           transform=ax.get_xaxis_transform())

            ax.scatter(k, c_k, s=50, color="white", edgecolors=ACCENT,
                       linewidths=1.2, zorder=5)
            ax.plot(k_plot, poly_curve, color=ACCENT, linewidth=2.0, zorder=4)
            ax.axhline(0, color=GRID, linewidth=0.9, linestyle="--", zorder=1)

            ax.text(0.97, 0.95, rf"$\hat{{\beta}} = {beta:.3e}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=GRID, alpha=0.9))

            ax.set_xlabel("Partialtonordnung  $k$", fontsize=12)
            ax.set_ylabel(r"Frequenzabweichung  $c_k$  in Hz", fontsize=12)
            ax.set_title(f"Inharmonizität — Frequenzabweichung  F0: {f0:.2f}",
                         fontsize=13, pad=10, fontweight="semibold")
            ax.set_xlim(0.5, K + 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaaaaa")
            ax.tick_params(colors="black", which="both")
            ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)

            legend_handles = [
                Line2D([0],[0], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor=ACCENT, markeredgewidth=1.2, markersize=7,
                       label=r"$c_k = f_k - k\,f_0$  (gemessen)"),
                Line2D([0],[0], color=ACCENT, linewidth=2,
                       label=r"Polynomfit  $a k^3 + b k + c$"),
            ]
            if len(k_invalid):
                legend_handles.append(
                    Line2D([0],[0], marker="|", color=DOT_NAN, markersize=9,
                           markeredgewidth=1.8, linestyle="none",
                           label="Ungültiger Partial (NaN)"))
            ax.legend(handles=legend_handles, facecolor="white", edgecolor=GRID,
                      fontsize=9.5, loc="upper left")

            fig.tight_layout()
            plt.show()

    # IQR Filter on betas directly from fitting
    betas = filter_betas(betas, 1e-3)  # high beta max -> capture everything first
    return betas


def _hz_to_bin_range(
    f_center: float,
    half_width_hz: float,
    W: int,
    sr: int,
    bin_nyquist: int,
) -> tuple[int, int]:
    """
    Translate (f_center ± half_width_hz) to a safe integer bin range.
    Guarantees at least 1 bin of search room and clamps to [0, bin_nyquist].
    """
    hz_per_bin = sr / W
    b_lo = max(int(np.floor((f_center - half_width_hz) / hz_per_bin)), 0)
    b_hi = min(int(np.ceil ((f_center + half_width_hz) / hz_per_bin)), bin_nyquist)

    if b_hi <= b_lo:
        b_lo = max(b_lo - 1, 0)
        b_hi = min(b_lo + 2, bin_nyquist)

    return b_lo, b_hi




# ── Constants (top of file or passed as params) ───────────────────────────
MAX_F0_JUMP_SEMITONES      = 0.5
MAX_PARTIAL_JUMP_SEMITONES = 1.0
SEARCH_SEMITONES           = 1.0   # half-width of peak search window around f̂ₖ

def semitone_gate(f0: float, semitones: float) -> float:
    """Max Hz deviation equivalent to `semitones` semitones at frequency f0."""
    return f0 * (2.0 ** (semitones / 12.0) - 1.0)


# ── Constants ─────────────────────────────────────────────────────────────
MAX_F0_JUMP_SEMITONES      = 0.5
MAX_PARTIAL_JUMP_SEMITONES = 1.0
SEARCH_SEMITONES_F0        = 0.5   # kept tight: f0 error multiplies into every partial
SEARCH_SEMITONES_PARTIAL   = 1.0   # scales in Hz with k; hard-capped at f0/2


def find_partials(
    fft_frames: np.ndarray,
    f0_guess: float,
    beta: float,
    sr: int,
    W: int,
    H: int,
    n_partials: int,
    amp_threshold: float = 0.005,
    partial_amp_threshold_dB: float = 0.0,
    note_name: str = "",
    plot: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:

    # ── Init ──────────────────────────────────
    n_frames, n_bins = fft_frames.shape
    bin_nyquist = W // 2

    search_windows: list = []

    fft_mag   = np.abs(fft_frames)
    freq_axis = np.arange(n_bins, dtype=float) * sr / W
    harmonic_orders = np.arange(1, n_partials + 1)

    partial_freqs = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_amps  = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_bins  = np.full((n_frames, n_partials), -1,    dtype=int)

    frame_max = fft_mag.max()
    if frame_max == 0.0:
        return partial_freqs, partial_amps, partial_bins, search_windows


    # ── Pre-compute instantaneous frequency ───────────────────────────────
    phase   = np.angle(fft_frames)
    k_bins  = np.arange(n_bins, dtype=float)
    omega_k = 2.0 * np.pi * k_bins / W

    delta_phi = (
        omega_k * H
        + np.mod(
            phase[1:, :] - phase[:-1, :] - omega_k * H + np.pi,
            2.0 * np.pi,
        )
        - np.pi
    )
    inst_freq = (delta_phi / (2.0 * np.pi * H)) * sr   # (n_frames-1, n_bins)

    # ── Normalised magnitude for peak-picking (Frame) ─────────────────────────────
    fft_norm = fft_mag / (fft_mag.max(axis=1, keepdims=True) + 1e-12)
    fft_norm[fft_norm < amp_threshold] = 0.0

    # # ── Normalised magnitude for peak-picking (NOte) ─────────────────────────────
    # fft_norm = fft_mag / (frame_max + 1e-12)
    # fft_norm[fft_norm < amp_threshold] = 0.0


    # ── Stage 1: refine f0 per frame ──────────────────────────────────────
    # Tight semitone window; f0 errors would otherwise scale into every partial.
    f0_search_hz = semitone_gate(f0_guess, SEARCH_SEMITONES_F0)
    b_lo_f0, b_hi_f0 = _hz_to_bin_range(f0_guess, f0_search_hz, W, sr, bin_nyquist)

    f0_per_frame = np.full(n_frames, np.nan, dtype=float)

    for t in range(1, n_frames):
        region = fft_norm[t, b_lo_f0-1: b_hi_f0 + 1] # TODO: extend search range down asymmetrically
        if region.size == 0:
            continue

        peaks_local, _ = find_peaks(region, height=0.0)
        if peaks_local.size == 0:
            continue

        best_local = peaks_local[np.argmax(region[peaks_local])]
        best_bin   = b_lo_f0-1 + best_local # TODO: hier auch -1

        if_est = inst_freq[t - 1, best_bin]
        candidate_f0 = (
            if_est
            if abs(if_est - freq_axis[best_bin]) < f0_search_hz
            else freq_axis[best_bin]
        )

        max_f0_jump = semitone_gate(f0_guess, MAX_F0_JUMP_SEMITONES)
        if not np.isnan(f0_per_frame[t - 1]):
            if abs(candidate_f0 - f0_per_frame[t - 1]) > max_f0_jump:
                f0_per_frame[t] = np.nan
                continue

        f0_per_frame[t] = candidate_f0


    # ── Stage 2: find each partial using the per-frame f0 ─────────────────
    W_AMP      = 2.0
    W_JUMP     = 8.0
    W_HARMONIC = 1.0

    for t in range(1, n_frames):
        f0_t = f0_per_frame[t]
        if np.isnan(f0_t):
            continue

        for p_idx, k in enumerate(harmonic_orders):
            f_k = k * f0_t * np.sqrt(1.0 + beta * float(k) ** 2)
            if f_k >= sr / 2.0 or np.isnan(f_k):
                break

            # Semitone window that grows in Hz with k, but never wider than f0/2
            # so adjacent partials cannot bleed into each other's search region.
            partial_search_hz = min(
                semitone_gate(f_k, SEARCH_SEMITONES_PARTIAL),
                f0_t / 2.0,
            )

            b_lo, b_hi = _hz_to_bin_range(f_k, partial_search_hz, W, sr, bin_nyquist)
            amp_region  = fft_norm[t, b_lo: b_hi + 1]
            freq_region = freq_axis[b_lo: b_hi + 1]

            if amp_region.size == 0:
                continue

            search_windows.append((t, f_k, freq_axis[b_lo], freq_axis[b_hi]))

            peaks_local, _ = find_peaks(amp_region)
            if peaks_local.size == 0:
                continue

            # Resolve each candidate via instantaneous frequency
            cand_freqs = np.empty(len(peaks_local), dtype=float)
            for i, pl in enumerate(peaks_local):
                bb     = b_lo + pl
                if_est = inst_freq[t - 1, bb]
                cand_freqs[i] = (
                    if_est
                    if abs(if_est - freq_axis[bb]) < partial_search_hz
                    else float(freq_region[pl])
                )
            cand_amps = amp_region[peaks_local]

            # ── Cost function ─────────────────────────────────────────────
            norm_amps = cand_amps / (cand_amps.max() + 1e-12)
            amp_cost  = 1.0 - norm_amps

            max_jump_hz = semitone_gate(f_k, MAX_PARTIAL_JUMP_SEMITONES)
            prev_freq   = partial_freqs[t - 1, p_idx]
            jump_cost   = (
                np.clip(np.abs(cand_freqs - prev_freq) / max_jump_hz, 0.0, 1.0)
                if not np.isnan(prev_freq)
                else np.zeros(len(peaks_local))
            )

            # Normalise harmonic deviation against the same window used for search
            harmonic_cost = np.clip(
                np.abs(cand_freqs - f_k) / (partial_search_hz + 1e-12), 0.0, 1.0
            )

            total_cost = W_AMP * amp_cost + W_JUMP * jump_cost + W_HARMONIC * harmonic_cost

            best_idx      = int(np.argmin(total_cost))
            best_bin      = b_lo + peaks_local[best_idx]
            detected_freq = cand_freqs[best_idx]

            # Hard gate: amplitude alone must not override a physically bad jump
            if not np.isnan(prev_freq) and abs(detected_freq - prev_freq) > max_jump_hz:
                continue

            partial_freqs[t, p_idx] = detected_freq
            partial_amps[t, p_idx]  = 20.0 * np.log10(float(cand_amps[best_idx]) + 1e-12)
            partial_bins[t, p_idx]  = best_bin

    # removes overlapping notes
    valid_mask = ~np.isnan(f0_per_frame)
    for t in np.where(valid_mask)[0]:
        neighbors = valid_mask[max(0, t - 2): t + 3]
        if neighbors.sum() < 2:
            f0_per_frame[t] = np.nan

    # ── Plot ─────────────────────────────────────────────────────────────────
    if False:
        times_frames = np.arange(n_frames) * (H / sr)
        fft_mag_db   = 20.0 * np.log10(fft_norm + 1e-12)

        fig, ax = plt.subplots(figsize=(14, 14))
        pcm = ax.pcolormesh(
            times_frames, freq_axis, fft_mag_db.T,
            shading="auto", cmap="magma",
            vmin=fft_mag_db.max() - 80,
            vmax=fft_mag_db.max(),
        )

        for p in range(partial_freqs.shape[1]):
            ax.plot(times_frames, partial_freqs[:, p],
                    linewidth=1.5, color="lime", alpha=0.85)

        half_frame = (H / sr) / 2.0
        for (t, f_k, f_lo, f_hi) in search_windows:
            t_center = times_frames[t]
            ax.hlines(
                [f_lo, f_hi],
                t_center - half_frame,
                t_center + half_frame,
                colors="cyan", linewidths=0.6, alpha=0.45,
            )

        ax.set_ylim(80, min(4000, sr / 2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        title_parts = []
        if note_name:
            title_parts.append(note_name)
        title_parts.append(f"f0={f0_guess:.1f} Hz")
        title_parts.append(f"β={beta:.2e}")
        title_parts.append(f"{n_partials} partials requested")
        ax.set_title("Inharmonic partial tracking — " + "  |  ".join(title_parts))

        plt.colorbar(pcm, ax=ax, label="Magnitude (dB)")
        plt.tight_layout()
        plt.show()

    return partial_freqs, partial_amps, partial_bins, search_windows



def process_single_file(args):
    filepath, beta_max, plot, threshold, W, H, audio_types = args

    print(f"Calculating Features {filepath}")

    # filepath = '/Users/simonbuechner/Documents/Studium/AKT/0MA/GuitarString-SourceSeparation/gse/noteData/GuitarSet/train/00_BN1-147-Gb_solo_track.pkl'

    # load track
    try:
        with open(filepath, "rb") as f:
            track = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except EOFError:
        print(f"File corrupted/empty: {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        return None

    for audio_type in audio_types:
        audio_filepath = track.audio_paths[audio_type]
        note_signal = pf.io.read_audio(audio_filepath)
        note_signal = pf.dsp.normalize(note_signal)
        sr = note_signal.sampling_rate

        for note in track.valid_notes:
            if not isinstance(note.features, dict):
                note.features = {}
            if audio_type not in note.features:
                note.features[audio_type] = Features()

            """ audio preprocessing """
            onset_sample = int(note.attributes.onset * sr)
            offset_sample = int(note.attributes.offset * sr)
            audio_data = note_signal.time  # shape: (6, N_samples)
            string_idx = note.attributes.string_index

            if audio_type == "hex_debleeded":
                note_audio = audio_data[string_idx, onset_sample:offset_sample]
            else:
                # mono or mixed signal — collapse to mono if needed
                if audio_data.ndim == 1:
                    note_audio = audio_data[onset_sample:offset_sample]
                else:
                    note_audio = audio_data[0, onset_sample:offset_sample]

            if note_audio is None:
                note.invalidate(FilterReason.NO_NOTE_AUDIO, step="find_partials")
                continue


            harmonic_audio = note_audio
            preprocessed_audio, window = note_audio_preprocess(harmonic_audio, W, H)

            if len(harmonic_audio) < W:
                note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
                continue

            if preprocessed_audio.ndim < 2:
                note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
                continue

            # rfft: shape (n_frames, W//2+1)
            fft_frames = np.fft.rfft(preprocessed_audio, axis=1)
            fret = note.attributes.fret
            f0 = note.attributes.pitch
            beta = 5e-6  # warm start
            n_iter = 50  # empirically 2–3 iterations usually suffice
            BETA_CONVERGENCE_MIN = 1e-5  # unterhalb davon ist Konvergenz bedeutungslos

            for i, iteration in enumerate(range(n_iter)):
                partial_freqs, partial_amps, partial_bins, search_windows = find_partials(
                    fft_frames=fft_frames,
                    f0_guess=f0,
                    beta=beta,
                    sr=sr,
                    W=W,
                    H=H,  # ← pass hop size
                    n_partials=25,
                    partial_amp_threshold_dB=threshold,
                    note_name=str(note.attributes.pitch),  # ← optional, used in title
                    plot=(string_idx == 0 or string_idx == 1),
                )


                t_frames = np.arange(partial_freqs.shape[0]) * (H / sr)
                # Store on the note object so the estimator can read them
                note.partials[audio_type] = Partials(
                    frametimes=t_frames,
                    frequencies=partial_freqs,
                    amplitudes=partial_amps,
                )

                # ---- estimate β from the current partial positions -----------------
                betas = estimate_inharmonicity_coefficient_all_frets(
                    note.partials[audio_type],)

                num_iterations = i

                if len(betas) == 0:
                    # note.invalidate(FilterReason.NO_BETAS, step="calculate_features")
                    break

                beta_new = kde_mode(np.array(betas))
                if np.isnan(beta_new):
                    beta_new = np.nanmedian(betas) # choose median if not enough values for kde



                if beta_new < 0:
                    # f0 ist wahrscheinlich überschätzt → leicht nach unten korrigieren
                    f0 *= 0.995  # z.B. ~0.87 Cent nach unten
                    beta = max(beta, 1e-7)  # Fallback, nicht 0

                if not np.isfinite(beta_new) or beta_new <= 0:
                    break  # or: keep current beta and break

                if abs(beta_new - beta) / beta_new < 0.001 and beta_new >= BETA_CONVERGENCE_MIN:
                    beta = beta_new
                    break

                beta = beta_new

            # scale beta down to open string
            betas0 = np.array(betas) * 2 ** (-fret / 6) if len(betas) > 0 else np.array([np.nan])
            # betas0 = filter_betas(betas0, 5e-4) # filter to realistic values
            beta0 = kde_mode(np.array(betas0))
            if np.isnan(beta0):
                beta0= np.nanmedian(betas0)

            # clean up distributions -> no 0.0
            if beta0 <=1e-7:
                beta0 = np.nan


            if note.partials[audio_type] is None:
                note.invalidate(FilterReason.NO_PARTIALS_FOUND, step="find_partials")
                continue

            # ── Final plot after convergence ──────────────────────────────────────────
            # if note.attributes.string_index == 0 or note.attributes.string_index == 1:
            if False:
                n_frames = fft_frames.shape[0]
                n_bins = fft_frames.shape[1]
                freq_axis = np.arange(n_bins, dtype=float) * sr / W

                fft_mag = np.abs(fft_frames)
                fft_norm = fft_mag / (fft_mag.max(axis=1, keepdims=True) + 1e-12)
                fft_norm_db = 20.0 * np.log10(fft_norm + 1e-12)

                times_frames = np.arange(n_frames) * (H / sr)

                fig, ax = plt.subplots(figsize=(14, 14))
                pcm = ax.pcolormesh(
                    times_frames, freq_axis, fft_norm_db.T,
                    shading="auto", cmap="magma",
                    vmin=fft_norm_db.max() - 80,
                    vmax=fft_norm_db.max(),
                )

                for p in range(partial_freqs.shape[1]):
                    ax.plot(times_frames, partial_freqs[:, p],
                            linewidth=1.5, color="lime", alpha=0.85)

                ax.set_ylim(80, min(8000, sr / 2))
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title(
                    f"Inharmonic partial tracking — {note.attributes.pitch}\n string: {note.attributes.string_index}"
                    f"  |  f0={f0:.1f} Hz  |  β={beta0:.2e}  |  {partial_freqs.shape[1]} partials \n Itration: {num_iterations}"
                )
                plt.colorbar(pcm, ax=ax, label="Magnitude (dB)")
                plt.tight_layout()
                plt.show()

            """ Features """
            # assign attributed f0 to features
            note.features[audio_type].f0 = note.attributes.pitch

            # scaled to open string beta0
            betas_measures = np.array([
                np.nanmean(betas0, axis=0),
                np.nanmedian(betas0, axis=0),
                np.nanstd(betas0, axis=0),
                np.nanvar(betas0, axis=0),
                np.nanmin(betas0, axis=0),
                np.nanmax(betas0, axis=0),
                stats.skew(betas0, nan_policy="omit"),
                stats.kurtosis(betas0, nan_policy="omit"),
                kde_mode(betas0),  # already filters NaNs internally
            ])


            sc_measures = spectral_centroid_feature(harmonic_audio, W, H, sr)
            note.features[audio_type].spectral_centroid = sc_measures

            amp_deviation_measures, amp_decay_coefficients = relative_amplitude_deviations(note.partials[audio_type]) # (25,)
            freq_deviation_measures = relative_freq_deviations(note.partials[audio_type], beta)

            valid_partial_flags = np.isfinite(note.partials[audio_type].amplitudes)  # (18, 25) bool
            valid_partial_measures = valid_partial_flags.mean(axis=0)  # (25,) float, range [0, 1]
            note.features[audio_type].valid_partials = valid_partial_measures


            # write into note.features
            note.features[audio_type].beta = beta0
            note.features[audio_type].betas_measures = betas_measures

            note.features[audio_type].rel_partial_amplitudes = amp_deviation_measures
            note.features[audio_type].rel_freq_deviations = freq_deviation_measures
            note.features[audio_type].amp_decay_coefficients = amp_decay_coefficients  # was on note, not note.features
            note.features[audio_type].fill_feature_vector()


    track.save(filepath)
    print(f"Saved {filepath}")
    return f"Success! Feature Vector (raw) calculated for: {os.path.basename(filepath)}"


def main(config):
    # read config
    W = config.getint('params', 'W')
    H = config.getint('params', 'H')
    beta_max = config.getfloat('params', 'beta_max')
    threshold = config.getint('params', 'threshold')
    plot = config.getboolean('params', 'plot')
    audio_types_raw = config.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]

    track_directory = config.get('paths', 'track_directory')

    print(W, H, beta_max, threshold)
    print(track_directory)

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if filename.endswith(".pkl")
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    args_list = [(fp, beta_max, plot, threshold, W,H, audio_types) for fp in filepaths]

    # Create pool and process files
    # num_processes = mp.cpu_count() - 1  # Leave one core free
    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.map(process_single_file, args_list)

    results = []
    for args in args_list:
        result = process_single_file(args)
        results.append(result)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")


if __name__ == "__main__":
    config = ConfigParser()
    config.read('configs/config_train_dev_GuitarSet.ini')

    main(config)