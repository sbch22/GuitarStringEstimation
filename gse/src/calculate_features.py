from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(''))

import multiprocessing as mp
import os
import pickle
from configparser import ConfigParser
import pyfar as pf
import librosa as lib
from scipy import stats
import numpy as np
from scipy.signal import find_peaks
import scipy

from utils.FeatureNote_dataclass import FilterReason
from utils.FeatureNote_dataclass import Features
import matplotlib.pyplot as plt


import inharmonic_partial_tracking


def note_audio_preprocess(audio, W, H, N_fft):
    """
    Preprocesses Audio:
    - Buffering
    - Windowing
    - Zero-padding to N_fft (if N_fft > W)
    """

    if N_fft is None:
        N_fft = W  # no extra padding by default

    audio_buffered = np.pad(audio, (0, W), mode="constant")
    audio_buffered = np.lib.stride_tricks.sliding_window_view(
        audio_buffered, window_shape=W
    )[::H]

    window = scipy.signal.windows.hann(W, sym=False)
    audio_windowed = audio_buffered * window

    # Zero-pad each frame to N_fft
    if N_fft > W:
        n_frames = audio_windowed.shape[0]
        padded = np.zeros((n_frames, N_fft), dtype=audio_windowed.dtype)
        padded[:, :W] = audio_windowed
        audio_windowed = padded

    return audio_windowed, window

""" FEATURES """
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

""" HELPERS """
def filter_betas(betas, beta_max):
    """Filter outliers from beta array using IQR method."""
    betas = np.asarray(betas, dtype=float).ravel()

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

def process_note(
    note,
    note_signal,
    audio_type,
    beta_min,
    beta_max,
    sr,
    W,
    H,
    threshold,
    plot,
):
    # --- init features ---
    if not isinstance(note.features, dict):
        note.features = {}
    if audio_type not in note.features:
        note.features[audio_type] = Features()

    # --- audio slicing ---
    onset_sample = int(note.attributes.onset * sr)
    offset_sample = int(note.attributes.offset * sr)
    audio_data = note_signal.time
    string_idx = note.attributes.string_index

    if audio_type == "hex_debleeded":
        note_audio = audio_data[string_idx, onset_sample:offset_sample]
    else:
        if audio_data.ndim == 1:
            note_audio = audio_data[onset_sample:offset_sample]
        else:
            note_audio = audio_data[0, onset_sample:offset_sample]

    if note_audio is None:
        note.invalidate(FilterReason.NO_NOTE_AUDIO, step="find_partials")
        return

    # --- preprocessing ---
    harmonic_audio = note_audio
    if len(harmonic_audio) < W:
        note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
        return

    # zero-padding to increase low-frequency candidate-picking
    N_fft = 16*W
    preprocessed_audio, window = note_audio_preprocess(harmonic_audio, W, H, N_fft)

    if preprocessed_audio.ndim < 2:
        note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
        return

    fft_frames = np.fft.rfft(preprocessed_audio, axis=1)

    # --- partial tracking ---
    partials, betas, beta, f0 = inharmonic_partial_tracking(
        fft_frames=fft_frames,
        f0=note.attributes.pitch,
        beta=beta_min, # initial lowest guess for all guitars from literature -> the closer the inital guess, the better.
        n_iter=50,
        sr=sr,
        N_fft=N_fft,
        H=H,
        threshold=threshold,
        note_name=str(note.attributes.pitch),
        plot=plot,
        beta_max=beta_max,
    )

    note.partials[audio_type] = partials

    if partials is None:
        note.invalidate(FilterReason.NO_PARTIALS_FOUND, step="find_partials")
        return

    # beta single number averaging
    betas = filter_betas(betas, beta_max)
    betas = np.array(betas)

    beta = kde_mode(betas)
    if np.isnan(beta):
        beta = np.nanmedian(betas)

    # --- optional plotting ---
    if plot:
        n_frames, n_bins = fft_frames.shape
        freq_axis = np.arange(n_bins) * sr / N_fft

        fft_mag = np.abs(fft_frames)
        fft_norm = fft_mag / (fft_mag.max() + 1e-12)
        fft_norm_db = 20 * np.log10(fft_norm)

        times_frames = np.arange(n_frames) * (H / sr)

        fig, ax = plt.subplots(figsize=(12, 8))
        pcm = ax.pcolormesh(
            times_frames, freq_axis, fft_norm_db.T,
            shading="auto", cmap="magma",
            vmin=fft_norm_db.max() - 80,
            vmax=fft_norm_db.max(),
        )

        partial_freqs = partials.frequencies

        for p in range(partial_freqs.shape[1]):
            ax.plot(times_frames, partial_freqs[:, p], color="lime", alpha=0.8, linewidth=2)

        ax.set_ylim(80, min(2000, sr / 2))
        ax.set_title(f"{note.attributes.pitch} Hz | β={beta:.2e} | String: {note.attributes.string_index}")
        plt.colorbar(pcm, ax=ax)
        plt.tight_layout()
        plt.show()

    # features
    note.features[audio_type].f0 = note.attributes.pitch


    if np.any(np.isfinite(betas)):
        betas_measures = np.array([
            np.nanmean(betas),
            np.nanmedian(betas),
            np.nanstd(betas),
            np.nanvar(betas),
            np.nanmin(betas),
            np.nanmax(betas),
            stats.skew(betas, nan_policy="omit"),
            stats.kurtosis(betas, nan_policy="omit"),
            kde_mode(betas),
        ])
    else:
        betas_measures = np.full(9, np.nan)

    sc_measures = spectral_centroid_feature(harmonic_audio, W, H, sr)

    amp_dev, amp_decay = relative_amplitude_deviations(partials)
    freq_dev = relative_freq_deviations(partials, beta)

    valid_flags = np.isfinite(partials.amplitudes)
    valid_measures = valid_flags.mean(axis=0)

    # --- assign ---
    feat = note.features[audio_type]
    feat.beta = beta
    feat.betas_measures = betas_measures
    feat.spectral_centroid = sc_measures
    feat.rel_partial_amplitudes = amp_dev
    feat.rel_freq_deviations = freq_dev
    feat.amp_decay_coefficients = amp_decay
    feat.valid_partials = valid_measures

    feat.fill_feature_vector()


def process_single_file(args):
    filepath, beta_min, beta_max, plot, threshold, W, H, audio_types = args

    print(f"Calculating Features {filepath}")

    with open(filepath, "rb") as f:
        track = pickle.load(f)

    for audio_type in audio_types:
        audio_filepath = track.audio_paths[audio_type]
        note_signal = pf.io.read_audio(audio_filepath)
        note_signal = pf.dsp.normalize(note_signal)
        sr = note_signal.sampling_rate

        for note in track.valid_notes:
            process_note(
                note=note,
                note_signal=note_signal,
                audio_type=audio_type,
                beta_min=beta_min,
                beta_max=beta_max,
                sr=sr,
                W=W,
                H=H,
                threshold=threshold,
                plot=plot,
            )

    track.save(filepath)
    print(f"Saved {filepath}")

    return f"Success! Feature Vector (raw) calculated for: {os.path.basename(filepath)}"


def main(config):
    # read config
    W = config.getint('params', 'W')
    H = config.getint('params', 'H')
    beta0_min = config.getfloat('params', 'beta0_min')
    beta0_max = config.getfloat('params', 'beta0_max')
    threshold = config.getint('params', 'threshold')
    plot = config.getboolean('params', 'plot')
    audio_types_raw = config.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]

    track_directory = config.get('paths', 'track_directory')

    beta_min = beta0_min # a fret only scales beta upwards
    beta_max = beta0_max * 2 ** (20/6) # 20th fret as large boundary

    print(W, H,beta_min, beta_max, threshold)
    print(track_directory)

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if filename.endswith(".pkl")
        if os.path.isfile(os.path.join(track_directory, filename))
        # and "solo" in filename
    ]

    args_list = [(fp, beta_min, beta_max, plot, threshold, W,H, audio_types) for fp in filepaths]

    # Create pool and process files
    num_processes = mp.cpu_count() - 1  # Leave one core free
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, args_list)

    """ Disable Multiprocessing"""
    # results = []
    # for args in args_list:
    #     result = process_single_file(args)
    #     results.append(result)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")


if __name__ == "__main__":
    config = ConfigParser()
    config.read('configs/config_train_dev_GuitarSet.ini')

    main(config)