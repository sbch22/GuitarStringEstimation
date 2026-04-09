"""
Inharmonic partial tracking for stiff-string instruments.

Provides iterative partial detection and inharmonicity coefficient (β)
estimation via physically-bounded asymmetric search windows and a
weighted least-squares model fit.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy import stats
import scipy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from gse.src.utils.FeatureNote_dataclass import Partials


# ── Initial Weights ───────────────────────────────────────────────────────────
W_AMP = 2.0
W_JUMP = 5.0
W_HARMONIC = 3.0


# ── KDE helper ────────────────────────────────────────────────────────────────
def kde_mode(data):
    """Return the mode of *data* estimated via Gaussian KDE."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 5:
        return np.nan
    if np.unique(data).size < 3:
        return np.nan
    if np.std(data) < 1e-10:
        return data.mean()

    try:
        kde = stats.gaussian_kde(data, bw_method="scott")
        x = np.linspace(data.min(), data.max(), 1000)
        return x[np.argmax(kde(x))]
    except np.linalg.LinAlgError:
        return np.nan


# ── Audio preprocessing ───────────────────────────────────────────────────────
def note_audio_preprocess(audio, W, H, N_fft):
    """
    Preprocesses audio: buffering, windowing, zero-padding to N_fft.
    """
    if N_fft is None:
        N_fft = W

    audio_buffered = np.pad(audio, (0, W), mode="constant")
    audio_buffered = np.lib.stride_tricks.sliding_window_view(
        audio_buffered, window_shape=W
    )[::H]

    window = scipy.signal.windows.hann(W, sym=False)
    audio_windowed = audio_buffered * window

    if N_fft > W:
        n_frames = audio_windowed.shape[0]
        padded = np.zeros((n_frames, N_fft), dtype=audio_windowed.dtype)
        padded[:, :W] = audio_windowed
        audio_windowed = padded

    return audio_windowed, window


# ── Frequency helpers ─────────────────────────────────────────────────────────
def quartertone_gate(f: float) -> float:
    """Max Hz deviation equivalent to a quarter-tone at frequency *f*."""
    semitones = 0.5
    return f * (2.0 ** (semitones / 12.0) - 1.0)


def _compute_inst_freq(
    fft_frames: np.ndarray,
    N_fft: int,
    H: int,
    sr: int,
) -> np.ndarray:
    """
    Instantaneous frequency via successive-frame phase difference.
    Returns shape (n_frames-1, n_bins) in Hz.
    """
    n_bins = fft_frames.shape[1]
    phase = np.angle(fft_frames)
    k_bins = np.arange(n_bins, dtype=float)
    omega_k = 2.0 * np.pi * k_bins / N_fft

    delta_phi = (
        omega_k * H
        + np.mod(
            phase[1:] - phase[:-1] - omega_k * H + np.pi,
            2.0 * np.pi,
        )
        - np.pi
    )
    return (delta_phi / (2.0 * np.pi * H)) * sr


def _resolve_freq_via_inst(
    peaks_local: np.ndarray,
    b_lo: int,
    freq_axis: np.ndarray,
    inst_freq_row: np.ndarray,
    search_hz: float,
) -> np.ndarray:
    """Replace bin-centre Hz with inst-freq estimate where it stays in-window."""
    resolved = np.empty(len(peaks_local), dtype=float)
    for i, pl in enumerate(peaks_local):
        bb = b_lo + pl
        if_est = inst_freq_row[bb]
        resolved[i] = (
            if_est
            if abs(if_est - freq_axis[bb]) < search_hz
            else freq_axis[bb]
        )
    return resolved


# ── f0 refinement ─────────────────────────────────────────────────────────────
def _refine_f0_per_frame(
    fft_norm: np.ndarray,
    inst_freq: np.ndarray,
    freq_axis: np.ndarray,
    f0_guess: float,
    N_fft: int,
    sr: int,
) -> np.ndarray:
    """Locate the f0 peak in a tight window per frame; reject large jumps."""
    n_frames = fft_norm.shape[0]
    bin_nyquist = N_fft // 2
    freq_res = sr / N_fft
    f0_search_hz = quartertone_gate(f0_guess)
    max_f0_jump = quartertone_gate(f0_guess)

    b_lo = max(1, int(np.floor((f0_guess - f0_search_hz) / freq_res)))
    b_hi = min(bin_nyquist, int(np.ceil((f0_guess + f0_search_hz) / freq_res)))

    f0_per_frame = np.full(n_frames, np.nan, dtype=float)

    for t in range(1, n_frames):
        region = fft_norm[t, b_lo: b_hi + 1]
        if region.size < 3:
            best_local = np.argmax(region)
            peaks_local = np.array([best_local])
        else:
            peaks_local, _ = find_peaks(region)
            if peaks_local.size == 0:
                continue

        best_local = peaks_local[np.argmax(region[peaks_local])]
        best_bin = b_lo + best_local
        if_est = inst_freq[t - 1, best_bin]
        candidate_f0 = (
            if_est
            if abs(if_est - freq_axis[best_bin]) < f0_search_hz
            else freq_axis[best_bin]
        )

        last_valid = next(
            (f0_per_frame[j] for j in range(t - 1, max(t - 5, -1), -1)
             if not np.isnan(f0_per_frame[j])),
            np.nan
        )
        if not np.isnan(last_valid):
            if abs(candidate_f0 - last_valid) > max_f0_jump:
                continue
        f0_per_frame[t] = candidate_f0

    return f0_per_frame


# ── Search-window bounds ──────────────────────────────────────────────────────

def _partial_search_bounds(k, f0, beta, beta_max, N_fft, sr, iteration):
    freq_res = sr / N_fft
    f_harm = k * f0

    if iteration == 0:
        beta_upper = beta_max
    else:
        beta_upper = min(beta * 3.0, beta_max)

    f_max = k * f0 * np.sqrt(1.0 + beta_upper * k ** 2)
    epsilon = max(freq_res, f0 * 0.02)
    f_lo = f_harm - epsilon
    f_hi = min(f_max, f_harm + f0 / 2.0)

    b_lo = max(1, int(np.floor(f_lo / freq_res)))
    b_hi = min(N_fft // 2, int(np.ceil(f_hi / freq_res)))
    return b_lo, b_hi, f_lo, f_hi


# ── Candidate scoring ─────────────────────────────────────────────────────────
def _score_candidates(
    cand_freqs: np.ndarray,
    cand_amps: np.ndarray,
    f_k: float,
    f_harm: float,
    partial_search_hz: float,
    prev_freq: float,
    iteration: int,
    subharm_penalty: float = 5.0,
    harmonic_growth: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-term cost with a corrected directional harmonic penalty.

    Sub-harmonic candidates (below f_harm) are penalised heavily because
    inharmonicity can only shift partials *upward*.
    """
    norm_amps = cand_amps / (cand_amps.max() + 1e-12)
    amp_cost = 1.0 - norm_amps

    max_jump_hz = quartertone_gate(f_k)
    jump_cost = (
        np.clip(np.abs(cand_freqs - prev_freq) / max_jump_hz, 0.0, 1.0)
        if not np.isnan(prev_freq)
        else np.zeros(len(cand_freqs))
    )

    below_harm = cand_freqs < f_harm

    sub_dev = np.abs(cand_freqs - f_harm) / (partial_search_hz + 1e-12)
    sub_cost = sub_dev * subharm_penalty

    norm_dev = np.abs(cand_freqs - f_k) / (partial_search_hz + 1e-12)

    harmonic_cost = np.clip(
        np.where(below_harm, sub_cost, norm_dev),
        0.0, 1.0,
    )

    w_harmonic_eff = W_HARMONIC * (1.0 + iteration * harmonic_growth)
    total_cost = W_AMP * amp_cost + W_JUMP * jump_cost + w_harmonic_eff * harmonic_cost

    return amp_cost, jump_cost, harmonic_cost, total_cost


# ── Core partial tracker ──────────────────────────────────────────────────────
def _track_partials(
    fft_norm: np.ndarray,
    inst_freq: np.ndarray,
    freq_axis: np.ndarray,
    f0_per_frame: np.ndarray,
    beta: float,
    beta_max: float,
    sr: int,
    N_fft: int,
    n_partials: int,
    harmonic_orders: np.ndarray,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Track partials using the asymmetric physical search window.
    """
    n_frames = fft_norm.shape[0]

    search_windows: list = []
    partial_freqs = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_amps = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_bins = np.full((n_frames, n_partials), -1, dtype=int)

    for t in range(1, n_frames):
        f0_t = f0_per_frame[t]
        if np.isnan(f0_t):
            continue

        for p_idx, k in enumerate(harmonic_orders):
            k_f = float(k)
            f_harm = k_f * f0_t
            f_k = k_f * f0_t * np.sqrt(1.0 + beta * k_f ** 2)
            if f_k >= sr / 2.0 or np.isnan(f_k):
                break

            b_lo, b_hi, f_lo, f_hi = _partial_search_bounds(
                k, f0_t, beta, beta_max, N_fft, sr, iteration
            )

            amp_region = fft_norm[t, b_lo: b_hi + 1]
            if amp_region.size == 0:
                continue

            n_bins_region = b_hi - b_lo + 1

            if n_bins_region <= 4:
                best_local = np.argmax(amp_region)
                if amp_region[best_local] == 0.0:
                    continue
                peaks_local = np.array([best_local])
            else:
                peaks_local, _ = find_peaks(amp_region)
                if peaks_local.size == 0:
                    continue
                peaks_local = peaks_local[amp_region[peaks_local] > 0.0]
                if peaks_local.size == 0:
                    continue

            search_windows.append((t, f_k, f_lo, f_hi))

            partial_search_hz = (f_hi - f_lo) / 2.0
            cand_freqs = _resolve_freq_via_inst(
                peaks_local, b_lo, freq_axis, inst_freq[t - 1], partial_search_hz,
            )
            cand_amps = amp_region[peaks_local]

            amp_cost, jump_cost, harmonic_cost, total_cost = _score_candidates(
                cand_freqs,
                cand_amps,
                f_k,
                f_harm,
                partial_search_hz,
                prev_freq=partial_freqs[t - 1, p_idx],
                iteration=iteration,
            )

            best_idx = int(np.argmin(total_cost))
            detected_freq = cand_freqs[best_idx]
            f_model_k = k_f * f0_t * np.sqrt(1.0 + beta * k_f ** 2)
            residual = abs(detected_freq - f_model_k)

            max_acceptable = f0_t * 0.5
            if residual > max_acceptable:
                continue

            max_jump_hz = quartertone_gate(f_k)
            prev_freq = partial_freqs[t - 1, p_idx]
            if not np.isnan(prev_freq) and abs(detected_freq - prev_freq) > max_jump_hz:
                continue

            partial_freqs[t, p_idx] = detected_freq
            partial_amps[t, p_idx] = 20.0 * np.log10(float(cand_amps[best_idx]) + 1e-12)
            partial_bins[t, p_idx] = b_lo + peaks_local[best_idx]

    return partial_freqs, partial_amps, partial_bins, search_windows


# ── Public entry: single-pass partial finder ──────────────────────────────────
def find_partials(
    fft_frames: np.ndarray,
    f0_guess: float,
    beta: float,
    sr: int,
    N_fft: int,
    H: int,
    n_partials: int,
    beta_max: float,
    iteration: int,
    amp_threshold: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Track inharmonic partials across STFT frames.

    The search window per partial is asymmetric and physically bounded:
      lower = k·f₀ − slack  (inharmonicity never shifts partials down)
      upper = k·f₀·√(1 + β_max·k²) + slack
    """
    n_frames, n_bins = fft_frames.shape
    if np.abs(fft_frames).max() == 0.0:
        empty = np.full((n_frames, n_partials), np.nan)
        return empty, empty, np.full((n_frames, n_partials), -1, dtype=int), []

    harmonic_orders = np.arange(1, n_partials + 1)
    fft_mag = np.abs(fft_frames)
    freq_axis = np.arange(n_bins, dtype=float) * sr / N_fft
    inst_freq = _compute_inst_freq(fft_frames, N_fft, H, sr)

    fft_norm = fft_mag / (fft_mag.max() + 1e-12)
    fft_norm[fft_norm < amp_threshold] = 0.0

    f0_per_frame = _refine_f0_per_frame(
        fft_norm, inst_freq, freq_axis, f0_guess, N_fft, sr
    )

    partial_freqs, partial_amps, partial_bins, search_windows = _track_partials(
        fft_norm, inst_freq, freq_axis,
        f0_per_frame, beta, beta_max, sr, N_fft,
        n_partials, harmonic_orders,
        iteration=iteration,
    )

    return partial_freqs, partial_amps, partial_bins, search_windows


# ── Inharmonicity coefficient estimation ──────────────────────────────────────
def estimate_inharmonicity_coefficient_all_frets(
    partials, beta_max, iteration, plot, plot_frame=10
):
    """
    Estimate β per frame by direct minimisation of the inharmonicity model.

    Model:  f_k = k · f₀ · √(1 + β · k²)

    Weights combine amplitude (louder → better freq estimate) and k²
    (higher partials carry more β information).
    """
    freqs = partials.frequencies
    amps_db = partials.amplitudes

    T, K = freqs.shape
    betas = np.full(T, np.nan)
    f0s = np.full(T, np.nan)

    k_full = np.arange(1, K + 1)

    for i in range(T):
        f_k = freqs[i]
        a_k_db = amps_db[i]

        valid = np.isfinite(f_k) & np.isfinite(a_k_db)
        if np.sum(valid) < 4:
            continue

        k = k_full[valid]
        f = f_k[valid]
        a_db = a_k_db[valid]

        if k.max() < 6:
            continue

        if valid[0] and k[0] == 1:
            f0_init = f[0]
        else:
            f0_init = np.median(f / k)

        f0_lo = f0_init * 2 ** (-1.0 / 12.0)
        f0_hi = f0_init * 2 ** (+1.0 / 12.0)

        w_amp = (10 ** (a_db / 20.0)) ** 2
        w_info = k ** 2
        w = w_amp * w_info
        w = w / (w.max() + 1e-12)

        def cost_joint(params):
            f0_trial, beta_trial = params
            f_model = k * f0_trial * np.sqrt(1.0 + beta_trial * k ** 2)
            return np.sum((w * (f - f_model)) ** 2)

        result = minimize(
            cost_joint, x0=[f0_init, 1e-5],
            bounds=[(f0_lo, f0_hi), (0, beta_max)],
            method='L-BFGS-B',
        )

        if not result.success:
            continue

        f0, beta = result.x

        f_model = k * f0 * np.sqrt(1.0 + beta * k ** 2)
        residuals = np.abs(f - f_model)

        max_plausible_error = f0 / 4.0
        inlier = residuals < max_plausible_error

        if inlier.sum() >= 4 and inlier.sum() < len(k):
            k2, f2, w2 = k[inlier], f[inlier], w[inlier]
            w2 = w2 / (w2.max() + 1e-12)

            def cost_clean(params):
                f0_t, beta_t = params
                f_m = k2 * f0_t * np.sqrt(1.0 + beta_t * k2 ** 2)
                return np.sum((w2 * (f2 - f_m)) ** 2)

            result2 = minimize(
                cost_clean, x0=[f0, beta],
                bounds=[(f0_lo, f0_hi), (0, beta_max)],
                method='L-BFGS-B',
            )
            if result2.success:
                f0, beta = result2.x

        if not (np.isfinite(beta) and beta > 0):
            continue

        betas[i] = beta
        f0s[i] = f0

        # ── Optional diagnostic plot ──────────────────────────────────
        if plot and plot_frame is not None and i == plot_frame:
            c_k = f - k * f0

            if 'inlier' not in dir():
                inlier = np.ones(len(k), dtype=bool)

            k_plot = np.linspace(1, K, 400)
            f_model_curve = k_plot * f0 * np.sqrt(1.0 + beta * k_plot ** 2)
            c_model_curve = f_model_curve - k_plot * f0

            ACCENT = "#1565c0"
            DOT_NAN = "#c62828"
            GRID = "#d0d0d0"

            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            k_invalid = k_full[~valid]
            if len(k_invalid):
                ax.scatter(
                    k_invalid, np.zeros(len(k_invalid)),
                    marker="|", s=140, linewidths=1.8, color=DOT_NAN,
                    alpha=0.85, zorder=3, clip_on=False,
                    transform=ax.get_xaxis_transform(),
                )

            ax.axhline(0, color=GRID, linewidth=0.9, linestyle="--", zorder=1)

            ax.plot(
                k_plot, c_model_curve,
                color=ACCENT, linewidth=2.0, zorder=4,
                label=rf"Modell $k f_0 \sqrt{{1 + \beta k^2}} - k f_0$",
            )

            sc = ax.scatter(
                k[inlier], c_k[inlier], c=a_db[inlier], cmap="viridis",
                s=60, edgecolors="black", linewidths=0.5, zorder=5,
            )

            if (~inlier).any():
                ax.scatter(
                    k[~inlier], c_k[~inlier],
                    marker="x", s=80, linewidths=2, color="#c62828",
                    zorder=6, label="Outlier (entfernt)",
                )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Amplitude (dBFS)")

            for ki, ci, wi in zip(k, c_k, w):
                ax.plot(
                    [ki, ki], [ci - wi * 2, ci + wi * 2],
                    color=ACCENT, alpha=0.3, linewidth=3, zorder=2,
                )

            ax.text(
                0.97, 0.95, rf"$\hat{{\beta}} = {beta:.3e}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="black",
                bbox=dict(
                    boxstyle="round,pad=0.4", facecolor="white",
                    edgecolor=GRID, alpha=0.9,
                ),
            )

            ax.set_xlabel("Partialtonordnung $k$", fontsize=12)
            ax.set_ylabel(r"Frequenzabweichung $c_k = f_k - k f_0$ (Hz)", fontsize=12)
            ax.set_title(
                f"Inharmonizitätsfit — Frame {i}, $f_0$ = {f0:.2f} Hz | i:{iteration}",
                fontsize=13, pad=10, fontweight="semibold",
            )
            ax.set_xlim(0.5, K + 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaaaaa")
            ax.tick_params(colors="black", which="both")
            ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)

            legend_handles = [
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor="white", markeredgecolor="black",
                       markeredgewidth=0.5, markersize=7,
                       label=r"$c_k$ (gemessen)"),
                Line2D([0], [0], color=ACCENT, linewidth=2,
                       label=rf"Modellfit $\beta = {beta:.2e}$"),
                Line2D([0], [0], color=ACCENT, linewidth=3, alpha=0.3,
                       label="Gewicht $w_i$ (Amplitude × $k^2$)"),
            ]
            if len(k_invalid):
                legend_handles.append(
                    Line2D([0], [0], marker="|", color=DOT_NAN,
                           markersize=9, markeredgewidth=1.8,
                           linestyle="none", label="Fehlender Partial"),
                )
            ax.legend(handles=legend_handles, facecolor="white",
                      edgecolor=GRID, fontsize=9.5, loc="upper left")

            fig.tight_layout()
            plt.show()

    return betas, f0s


# ── Public entry: iterative tracker with β convergence ────────────────────────
def inharmonic_partial_tracker(
    fft_frames, f0, beta, n_iter,
    sr, N_fft, H, beta_max, threshold, plot, note_name=None,
    beta_bump_factor: float = 10.0,
    max_bumps: int = 3,
):
    """
    Run iterative partial tracking with β refinement until convergence.

    Returns (partials, betas, beta, f0).
    """
    last_partials = None
    n_bumps = 0

    for i in range(n_iter):
        iteration = i

        partial_freqs, partial_amps, partial_bins, search_windows = find_partials(
            fft_frames=fft_frames,
            f0_guess=f0,
            beta=beta,
            sr=sr, N_fft=N_fft, H=H,
            n_partials=25,
            beta_max=beta_max,
            iteration=iteration,
        )

        t_frames = np.arange(partial_freqs.shape[0]) * (H / sr)
        current_partials = Partials(
            frametimes=t_frames,
            frequencies=partial_freqs,
            amplitudes=partial_amps,
        )
        last_partials = current_partials

        betas, f0s = estimate_inharmonicity_coefficient_all_frets(
            current_partials, beta_max=beta_max, iteration=iteration, plot=plot,
        )

        if np.all(np.isnan(betas)) or len(betas) == 0:
            if n_bumps < max_bumps:
                beta = min(beta * beta_bump_factor, beta_max)
                n_bumps += 1
                continue
            else:
                break

        beta_new = kde_mode(np.array(betas))
        if np.isnan(beta_new):
            beta_new = np.nanmedian(betas)

        f0_new = np.nanmedian(f0s)
        if np.isfinite(f0_new):
            f0 = f0_new

        if not np.isfinite(beta_new) or beta_new < 0:
            if n_bumps < max_bumps:
                beta = min(beta * beta_bump_factor, beta_max)
                n_bumps += 1
                continue
            else:
                break

        if abs(beta_new - beta) / (beta_new + 1e-12) < 0.01:
            beta = beta_new
            break

        beta = beta_new

    return last_partials, betas, beta, f0
