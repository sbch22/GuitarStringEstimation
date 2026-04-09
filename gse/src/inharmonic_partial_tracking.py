from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(''))

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
from gse.src.utils.FeatureNote_dataclass import Partials
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

def estimate_inharmonicity_coefficient_all_frets(partials, beta_max, iteration, plot, plot_frame=10):
    """
    Schätzt β pro Frame durch direkte Minimierung des Inharmonizitätsmodells.

    Fitting-Logik:
    ─────────────
    Das physikalische Modell für den k-ten Partialton einer steifen Saite ist:

        f_k = k · f₀ · √(1 + β · k²)

    Gesucht wird das β ∈ [0, β_max], das die gewichtete Summe der
    quadratischen Residuen minimiert:

        β* = argmin  Σ  [ w_i · (f_k_i − k_i · f₀ · √(1 + β · k_i²)) ]²
             β∈[0,β_max]

    Die Gewichte w kombinieren zwei Faktoren:
      1. Amplitude (linear, ^1.5): Lautere Partialtöne haben zuverlässigere
         Frequenzschätzungen im STFT.
      2. k²: Höhere Partialtöne tragen mehr β-Information, da der
         inharmonische Shift mit k² wächst. Bei k=2 verschiebt β=1e-4
         nur ~0.07 Hz, bei k=10 schon ~8 Hz.

    Im Gegensatz zum vorherigen Taylor-Polynomfit (c_k ≈ a·k³ + b·k + c,
    β = 2a/(f₀+b)) hat dieser Ansatz zwei Vorteile:
      - β ist strukturell auf [0, β_max] beschränkt (kann nie negativ werden)
      - Nur 1 freier Parameter statt 3 → stabiler bei wenigen Partialtönen
    """
    freqs = partials.frequencies   # (T, K)
    amps_db = partials.amplitudes  # (T, K) in dBFS

    T, K = freqs.shape
    betas = np.full(T, np.nan)
    f0s = np.full(T, np.nan)  # ← neu

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

        # f0 darf sich nur wenig bewegen: ±1 Halbton
        f0_lo = f0_init * 2 ** (-1.0 / 12.0)
        f0_hi = f0_init * 2 ** (+1.0 / 12.0)


        # --- Gewichtung: Amplitude × Informationsgehalt ---
        w_amp = (10 ** (a_db / 20.0)) **2
        w_info = k ** 2
        w = w_amp * w_info
        w = w / (w.max() + 1e-12)


        def cost_joint(params):
            f0_trial, beta_trial = params
            f_model = k * f0_trial * np.sqrt(1.0 + beta_trial * k ** 2)
            return np.sum((w * (f - f_model)) ** 2)

        result = minimize(cost_joint, x0=[f0_init, 1e-5],
                          bounds=[(f0_lo, f0_hi), (0, beta_max)],
                          method='L-BFGS-B')

        if not result.success:
            continue

        # Right after the first result.x:
        f0, beta = result.x

        f_model = k * f0 * np.sqrt(1.0 + beta * k ** 2)
        residuals = np.abs(f - f_model)

        # A correctly identified partial can't be off by more than
        # half the distance to its neighbor — otherwise it's the
        # wrong peak entirely.
        max_plausible_error = f0 / 4.0
        inlier = residuals < max_plausible_error

        if inlier.sum() >= 4 and inlier.sum() < len(k):
            k2, f2, w2 = k[inlier], f[inlier], w[inlier]
            w2 = w2 / (w2.max() + 1e-12)

            def cost_clean(params):
                f0_t, beta_t = params
                f_m = k2 * f0_t * np.sqrt(1.0 + beta_t * k2 ** 2)
                return np.sum((w2 * (f2 - f_m)) ** 2)

            result2 = minimize(cost_clean, x0=[f0, beta],
                               bounds=[(f0_lo, f0_hi), (0, beta_max)],
                               method='L-BFGS-B')
            if result2.success:
                f0, beta = result2.x

        if not (np.isfinite(beta) and beta > 0):
            continue

        betas[i] = beta
        f0s[i] = f0

        # ── Plot ──────────────────────────────────────────────────────
        if plot and plot_frame is not None and i == plot_frame:
            c_k = f - k * f0

            # ── Mark outliers vs inliers ──
            # 'inlier' mask is from the residual gating above;
            # if no refit happened, all points are inliers
            if 'inlier' not in dir():
                inlier = np.ones(len(k), dtype=bool)

            k_plot = np.linspace(1, K, 400)
            f_model_curve = k_plot * f0 * np.sqrt(1.0 + beta * k_plot ** 2)
            c_model_curve = f_model_curve - k_plot * f0

            ACCENT  = "#1565c0"
            DOT_NAN = "#c62828"
            GRID    = "#d0d0d0"

            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            # Ungültige Partiale als Rug-Markierung
            k_invalid = k_full[~valid]
            if len(k_invalid):
                ax.scatter(
                    k_invalid, np.zeros(len(k_invalid)),
                    marker="|", s=140, linewidths=1.8, color=DOT_NAN,
                    alpha=0.85, zorder=3, clip_on=False,
                    transform=ax.get_xaxis_transform(),
                )

            ax.axhline(0, color=GRID, linewidth=0.9, linestyle="--", zorder=1)

            # Modellkurve: f_k - k·f₀ für geschätztes β
            ax.plot(
                k_plot, c_model_curve,
                color=ACCENT, linewidth=2.0, zorder=4,
                label=rf"Modell $k f_0 \sqrt{{1 + \beta k^2}} - k f_0$",
            )

            # Inlier points (color-coded by amplitude, as before)
            sc = ax.scatter(
                k[inlier], c_k[inlier], c=a_db[inlier], cmap="viridis",
                s=60, edgecolors="black", linewidths=0.5, zorder=5,
            )

            # Outlier points — red X markers
            if (~inlier).any():
                ax.scatter(
                    k[~inlier], c_k[~inlier],
                    marker="x", s=80, linewidths=2, color="#c62828",
                    zorder=6, label="Outlier (entfernt)",
                )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Amplitude (dBFS)")

            # Gewichte als Balkenbreite visualisieren
            for ki, ci, wi in zip(k, c_k, w):
                ax.plot(
                    [ki, ki], [ci - wi * 2, ci + wi * 2],
                    color=ACCENT, alpha=0.3, linewidth=3, zorder=2,
                )

            # β-Annotation
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


""" PARTIAL TRACKING """
def quartertone_gate(f: float) -> float:
    """Max Hz deviation equivalent to quartertones at frequency f."""
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
            # Zu wenig Bins für find_peaks → nimm das Maximum direkt
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

def _partial_search_bounds(k, f0, beta, beta_max, N_fft, sr, iteration):
    freq_res = sr / N_fft
    f_harm = k * f0

    # iteration 0: use beta_max (explore)
    # later: use current beta + shrinking margin
    if iteration == 0:
        beta_upper = beta_max
    else:
        # allow 3× current beta as headroom, but never exceed beta_max
        beta_upper = min(beta * 3.0, beta_max)

    f_max = k * f0 * np.sqrt(1.0 + beta_upper * k**2)
    epsilon = max(freq_res, f0 * 0.02)
    f_lo = f_harm - epsilon
    f_hi = min(f_max, f_harm + f0 / 2.0)

    b_lo = max(1, int(np.floor(f_lo / freq_res)))
    b_hi = min(N_fft // 2, int(np.ceil(f_hi / freq_res)))
    return b_lo, b_hi, f_lo, f_hi

# ── Initial Weights for the cost function ─────────────────────────────────────────
W_AMP = 2.0
W_JUMP = 5.0
W_HARMONIC = 3.0

def _score_candidates(
        cand_freqs: np.ndarray,
        cand_amps: np.ndarray,
        f_k: float,
        f_harm: float,
        partial_search_hz: float,
        prev_freq: float,
        iteration: int,
        subharm_penalty: float = 5.0,   # ← tune this; 5× works well empirically
        harmonic_growth: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-term cost function with a corrected directional harmonic penalty.

    Key fix: the sub-harmonic penalty is applied to the *unclipped* deviation
    and measured from f_harm (the β=0 anchor), not from f_k.  The old code
    clipped raw_harmonic_cost to [0,1] before multiplying — meaning candidates
    already at max cost were unaffected and the ×2 did nothing.

    Penalty tiers:
      cand < f_harm   →  deviation from f_harm × subharm_penalty  (physically impossible)
      f_harm ≤ cand   →  deviation from f_k   × 1.0               (normal inharmonic window)
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

    # Sub-harmonic branch: measure from f_harm and apply heavy penalty
    sub_dev = np.abs(cand_freqs - f_harm) / (partial_search_hz + 1e-12)
    sub_cost = sub_dev * subharm_penalty  # intentionally unclipped before merge

    # Normal branch: deviation from current β-shifted prediction
    norm_dev = np.abs(cand_freqs - f_k) / (partial_search_hz + 1e-12)

    # Merge and clip once at the end
    harmonic_cost = np.clip(
        np.where(below_harm, sub_cost, norm_dev),
        0.0, 1.0,
    )

    w_harmonic_eff = W_HARMONIC * (1.0 + iteration * harmonic_growth)
    total_cost = W_AMP * amp_cost + W_JUMP * jump_cost + w_harmonic_eff * harmonic_cost

    return amp_cost, jump_cost, harmonic_cost, total_cost

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

    beta_max is now a required parameter — it drives the upper search bound
    and tightens naturally as beta converges toward beta_max from below.
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
            f_harm = k_f * f0_t  # β = 0
            f_k = k_f * f0_t * np.sqrt(1.0 + beta * k_f ** 2)  # current β
            if f_k >= sr / 2.0 or np.isnan(f_k):
                break

            # ── Asymmetric physical bounds ─────────────────────────────
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
                # Also filter peaks that sit on the zeroed floor
                peaks_local = peaks_local[amp_region[peaks_local] > 0.0]
                if peaks_local.size == 0:
                    continue

            search_windows.append((t, f_k, f_lo, f_hi))

            # Use the window half-width for inst-freq gating
            partial_search_hz = (f_hi - f_lo) / 2.0
            cand_freqs = _resolve_freq_via_inst(
                peaks_local, b_lo, freq_axis, inst_freq[t - 1], partial_search_hz,
            )
            cand_amps = amp_region[peaks_local]

            amp_cost, jump_cost, harmonic_cost, total_cost = _score_candidates(
                cand_freqs,
                cand_amps,
                f_k,
                f_harm,  # ← directional penalty pivot
                partial_search_hz,
                prev_freq=partial_freqs[t - 1, p_idx],
                iteration=iteration,
            )

            best_idx = int(np.argmin(total_cost))
            # after best_idx selection, before writing to partial_freqs
            detected_freq = cand_freqs[best_idx]
            f_model_k = k_f * f0_t * np.sqrt(1.0 + beta * k_f ** 2)
            residual = abs(detected_freq - f_model_k)

            # half the distance to the neighboring harmonic
            max_acceptable = f0_t * 0.5
            if residual > max_acceptable:
                continue  # skip — leave NaN

            # hard gate for tonal confusion
            max_jump_hz = quartertone_gate(f_k)
            prev_freq = partial_freqs[t - 1, p_idx]
            if not np.isnan(prev_freq) and abs(detected_freq - prev_freq) > max_jump_hz:
                continue

            partial_freqs[t, p_idx] = detected_freq
            partial_amps[t, p_idx] = 20.0 * np.log10(float(cand_amps[best_idx]) + 1e-12)
            partial_bins[t, p_idx] = b_lo + peaks_local[best_idx]

    return partial_freqs, partial_amps, partial_bins, search_windows


def find_partials(
        fft_frames: np.ndarray,
        f0_guess: float,
        beta: float,
        sr: int,
        N_fft: int,
        H: int,
        n_partials: int,
        beta_max: float,  # ← upper physical bound
        iteration: int,
        amp_threshold: float = 0.005, # 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Track inharmonic partials across STFT frames.

    The search window per partial is now *asymmetric and physically bounded*:

      lower =  k·f₀  −  F0_ERROR_SEMITONES slack
               (inharmonicity never shifts partials down)
      upper =  k·f₀·√(1 + β_max·k²)  +  BETA_MAX_SLACK_SEMITONES slack
               (literature β_max ≈ 5e-4 caps the upward shift)

    As beta converges during the outer iteration loop the upper bound
    tightens automatically, making later iterations more discriminating.
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

    f0_per_frame = _refine_f0_per_frame(fft_norm, inst_freq, freq_axis, f0_guess, N_fft, sr)

    partial_freqs, partial_amps, partial_bins, search_windows = _track_partials(
        fft_norm, inst_freq, freq_axis,
        f0_per_frame, beta, beta_max, sr, N_fft,
        n_partials, harmonic_orders,
        iteration=iteration,
    )

    return partial_freqs, partial_amps, partial_bins, search_windows

def inharmonic_partial_tracking(
    fft_frames, f0, beta, n_iter,
    sr, N_fft, H, beta_max, threshold, plot, note_name=None,
    beta_bump_factor: float = 10.0,    # beta bump factor
    max_bumps: int = 3,
):
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

        # ---- Beta schätzen ----
        betas, f0s = estimate_inharmonicity_coefficient_all_frets(current_partials, beta_max=beta_max, iteration=iteration, plot=plot)

        if np.all(np.isnan(betas)) or len(betas) == 0:  # true when every frame failed
            # no partials found -> beta bump
            if n_bumps < max_bumps:
                beta = min(beta * beta_bump_factor, beta_max)
                n_bumps += 1
                continue
            else:
                break

        beta_new = kde_mode(np.array(betas))
        if np.isnan(beta_new):
            beta_new = np.nanmedian(betas)

        # ---- f0 updaten ---- from model fit
        f0_new = np.nanmedian(f0s)
        if np.isfinite(f0_new):
            f0 = f0_new

        if not np.isfinite(beta_new) or beta_new < 0:
            # Negatives/ungültiges Beta → Startwert hochsetzen, neu starten
            if n_bumps < max_bumps:
                beta = min(beta * beta_bump_factor, beta_max)
                n_bumps += 1
                continue
            else:
                break

        # Convergence?
        if abs(beta_new - beta) / (beta_new + 1e-12) < 0.01: #:
            beta = beta_new
            break

        beta = beta_new

    return last_partials, betas, beta, f0