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

def spectral_centroid_feature(preprocessed_audio, W, H, sr):
    """ Spectral Centroid """
    sc = lib.feature.spectral_centroid(y=preprocessed_audio, sr=sr, n_fft=W, hop_length=H)

    median = np.nanmedian(
        sc, axis=1
    )
    mean = np.nanmean(
        sc,
        axis=1
    )
    min = np.nanmin(
        sc,
        axis=1
    )
    max = np.nanmax(
        sc,
        axis=1
    )
    std = np.nanstd(
        sc,
        axis=1
    )  # (K-1,)
    var = np.nanvar(
        sc,
        axis=1
    )  # (K-1,)

    sc_measures = np.array([
        median,
        mean,
        min,
        max,
        std,
        var,
        stats.skew(sc, axis=1, nan_policy="omit"),
        stats.kurtosis(sc, axis=1, nan_policy="omit"),
        np.apply_along_axis(kde_mode, axis=1, arr=sc),
    ])
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
    rel_amps = amps / amps[:, 0:1]

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
        kde = stats.gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        return x[np.argmax(kde(x))]
    except np.linalg.LinAlgError:
        return np.nan


def estimate_inharmonicity_coefficient_all_frets(partials, fret, min_partials=4):
    freqs = partials.frequencies  # (T, K)
    T, K = freqs.shape

    betas = np.full(T, np.nan)
    betas0 = np.full(T, np.nan)

    k_full = np.arange(1, K + 1)

    for i in range(T):
        f_k = freqs[i]

        valid = np.isfinite(f_k)
        if np.sum(valid) < min_partials:
            continue

        k = k_full[valid]
        f = f_k[valid]

        if k[0] != 1:
            continue  # can't reliably estimate f0 without the fundamental
        f0 = f[0]

        # Eq. (15): ck
        c_k = f - k * f0

        # Eq. (16): polynomial fit
        X = np.vstack([k**3, k, np.ones_like(k)]).T

        try:
            a, b, c = np.linalg.lstsq(X, c_k, rcond=None)[0]

            # Eq. (17): beta
            beta = 2 * a / (f0 + b)

            # eig. beta0
            beta0 = beta * 2**(-fret/6)

            # sanity checks
            if np.isfinite(beta) and beta > 0:
                betas[i] = beta
            # sanity checks
            if np.isfinite(beta0) and beta0 > 0:
                betas0[i] = beta0

        except np.linalg.LinAlgError:
            continue

    return betas, betas0


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


def find_partials(
    fft_frames: np.ndarray,
    f0_guess: float,
    beta: float,
    sr: int,
    W: int,
    n_partials: int,
    search_hz: float = 10.0,
    amp_threshold: float = 0.1,
    partial_amp_threshold_dB: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.array]:

    n_frames, n_bins = fft_frames.shape
    bin_nyquist = W // 2

    search_windows = []

    fft_mag   = np.abs(fft_frames)
    freq_axis = np.arange(n_bins, dtype=float) * sr / W
    harmonic_orders = np.arange(1, n_partials + 1)

    partial_freqs = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_amps  = np.full((n_frames, n_partials), np.nan, dtype=float)
    partial_bins  = np.full((n_frames, n_partials), -1,    dtype=int)

    frame_max = fft_mag.max()
    if frame_max == 0.0:
        return partial_freqs, partial_amps, partial_bins   # nothing to find

    fft_norm = fft_mag / (fft_mag.max(axis=1, keepdims=True) + 1e-12)
    fft_norm[fft_norm < amp_threshold] = 0.0

    # ------------------------------------------------------------------
    # Stage 1 — refine f0 per frame
    # ------------------------------------------------------------------
    b_lo_f0, b_hi_f0 = _hz_to_bin_range(f0_guess, search_hz, W, sr, bin_nyquist)
    f0_per_frame = np.full(n_frames, np.nan, dtype=float)

    for t in range(n_frames):
        region = fft_norm[t, b_lo_f0 : b_hi_f0 + 1]
        if region.size == 0:
            continue

        peaks_local, _ = find_peaks(region, height=0.0)
        if peaks_local.size == 0:
            continue

        best_local = peaks_local[np.argmax(region[peaks_local])]
        f0_per_frame[t] = freq_axis[b_lo_f0 + best_local]

    # ------------------------------------------------------------------
    # Stage 2 — find each partial using the per-frame f0
    # ------------------------------------------------------------------
    for t in range(n_frames):                          # ← single loop, correct indent
        f0_t = f0_per_frame[t]
        if np.isnan(f0_t):
            continue

        for p_idx, k in enumerate(harmonic_orders):
            f_k = k * f0_t * np.sqrt(1.0 + beta * float(k) ** 2)
            if f_k >= sr / 2.0:
                break
            elif np.isnan(f_k):
                continue


            b_lo, b_hi  = _hz_to_bin_range(f_k, search_hz, W, sr, bin_nyquist)
            amp_region  = fft_norm[t, b_lo : b_hi + 1]
            freq_region = freq_axis[b_lo : b_hi + 1]

            if amp_region.size == 0:
                continue

            peaks_local, _ = find_peaks(amp_region, height=0.0)

            if peaks_local.size == 0:
                continue

            # choose peak closest to predicted frequency
            best_local = peaks_local[np.argmax(amp_region[peaks_local])]

            partial_freqs[t, p_idx] = float(freq_region[best_local])
            partial_amps [t, p_idx] = 20.0 * np.log10(float(amp_region[best_local]) + 1e-12)
            partial_bins [t, p_idx] = b_lo + best_local

            search_windows.append((t, f_k, freq_axis[b_lo], freq_axis[b_hi]))

    return partial_freqs, partial_amps, partial_bins, search_windows



def process_single_file(args):
    filepath, beta_max, plot, threshold, W, H, audio_types = args

    print(f"Calculating Features {filepath}")

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
            # extract note audio from
            onset_sample = int(note.attributes.onset * sr)
            offset_sample = int(note.attributes.offset * sr)

            audio_data = note_signal.time  # shape: (6, N_samples)

            if audio_type == "hex_debleeded":
                string_idx = note.attributes.string_index
                if string_idx is None or not (0 <= string_idx < audio_data.shape[0]):
                    note.invalidate(FilterReason.NO_STRING, step="find_partials")
                    continue
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

            # call extraction of harmonic audio
            harmonic_audio = note_audio
            # harmonic_audio, intra_onsets = extract_harmonic_note_audio(note_audio, W, H, sr, plot)
            if harmonic_audio is None:
                note.invalidate(FilterReason.NO_HARMONIC_AUDIO, step="find_partials")
                continue
            preprocessed_audio, window = note_audio_preprocess(harmonic_audio, W, H)

            if preprocessed_audio.ndim < 2:
                note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
                continue

            # rfft: shape (n_frames, W//2+1)
            fft_frames = np.fft.rfft(preprocessed_audio, axis=1)

            f0 = note.attributes.pitch
            beta = 0.0  # start with pure-harmonic assumption
            n_iter = 50  # empirically 2–3 iterations usually suffice

            for i, iteration in enumerate(range(n_iter)):
                # ---- find peaks in spectrum ----------------------------------------
                partial_freqs, partial_amps, partial_bins, search_windows = find_partials(
                    fft_frames=fft_frames,
                    f0_guess=f0,
                    beta=beta,
                    sr=sr,  # your sample-rate constant
                    W=W,
                    n_partials=50,
                    search_hz=f0/4,
                    partial_amp_threshold_dB=threshold,
                )


                t_frames = np.arange(partial_freqs.shape[0]) * (H / sr)
                # Store on the note object so the estimator can read them
                note.partials[audio_type] = Partials(
                    frametimes=t_frames,
                    frequencies=partial_freqs,
                    amplitudes=partial_amps,
                )

                # ---- estimate β from the current partial positions -----------------
                betas, betas0 = estimate_inharmonicity_coefficient_all_frets(
                    note.partials[audio_type],
                    note.attributes.fret,
                )
                if len(betas) == 0:
                    note.invalidate(FilterReason.NO_BETAS, step="calculate_features")
                    break

                beta_new = np.nanmedian(betas)
                if not np.isfinite(beta_new) or beta_new <= 0:
                    break  # or: keep current beta and break

                if abs(beta_new - beta) / beta_new < 0.001:
                    break

                beta = beta_new



                """ Plot every iteration"""
                if True:
                    # --- Spectrogram with partials overlay ---
                    fft_mag = np.abs(np.fft.rfft(preprocessed_audio, axis=1))
                    fft_mag_db = 20 * np.log10(fft_mag + 1e-12)

                    # Drop first FFT frame to match inst_freq length
                    fft_mag_db_if = fft_mag_db[0:]
                    times_if = np.arange(fft_mag_db_if.shape[0]) * (H / sr)

                    freqs = np.fft.rfftfreq(W, 1 / sr)

                    plt.figure(figsize=(14, 12))

                    pcm = plt.pcolormesh(
                        times_if,
                        freqs,
                        fft_mag_db_if.T,
                        shading="auto",
                        cmap="magma",
                        vmin=threshold,
                        vmax=2,
                    )

                    # Overlay partials
                    for p in range(partial_freqs.shape[1]):
                        plt.plot(times_if, partial_freqs[:, p], linewidth=2.5, color='g')

                    for (t, f_k, f_lo, f_hi) in search_windows:
                        plt.hlines(
                            [f_lo, f_hi],
                            times_if[t] - H / sr / 2,
                            times_if[t] + H / sr / 2,
                            color="cyan",
                            alpha=0.6
                        )

                    # plt.yscale("log")
                    plt.ylim(80, 8000)
                    plt.colorbar(pcm, label="Magnitude (dB)")
                    plt.title(
                        f"Spectrogram with Extracted Partials – "
                        f"{note.attributes.pitch:.2f} Hz, String: {note.attributes.string_index}\n"
                        f"Beta: {beta:.2f}"
                        f"Iteration: {i}"
                    )
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency (Hz)")
                    # plt.legend(ncol=4, fontsize=9)
                    plt.tight_layout()
                    plt.show()


            # betas = filter_betas(betas, beta_max)
            # if len(betas) == 0:
            #     note.invalidate(FilterReason.NO_BETAS_AFTER_FILTER, step="calculate_features")
            #     continue

            """ Features """
            # assign attributed f0 to features
            note.features[audio_type].f0 = note.attributes.pitch

            betas_measures = np.array([
                np.nanmean(betas, axis=0),
                np.nanmedian(betas, axis=0),
                np.nanstd(betas, axis=0),
                np.nanvar(betas, axis=0),
                np.nanmin(betas, axis=0),
                np.nanmax(betas, axis=0),
                stats.skew(betas, nan_policy="omit"),
                stats.kurtosis(betas, nan_policy="omit"),
                kde_mode(betas),  # already filters NaNs internally
            ])


            sc_measures = spectral_centroid_feature(preprocessed_audio, W, H, sr)
            note.features[audio_type].spectral_centroid = sc_measures


            amp_deviation_measures, amp_decay_coefficients = relative_amplitude_deviations(note.partials[audio_type]) # (25,)
            freq_deviation_measures = relative_freq_deviations(note.partials[audio_type], beta)

            # valid_slopes = np.isfinite(amp_decay_coefficients)
            valid_partials = np.isfinite(amp_decay_coefficients)#.astype(int)

            # write into note.features
            note.features[audio_type].beta = beta
            note.features[audio_type].betas_measures = betas_measures

            note.features[audio_type].valid_partials = valid_partials
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
    config.read('configs/config_train_single_note_IDMT.ini')

    main(config)