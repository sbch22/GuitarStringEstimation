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

from utils.FeatureNote_dataclass import FilterReason
from utils.FeatureNote_dataclass import Features


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


def estimate_inharmonicity_coefficient_all_frets(partials, fret, min_partials=6):
    freqs = partials.frequencies  # (T, K)
    T, K = freqs.shape

    betas = np.full(T, np.nan)
    k_full = np.arange(1, K + 1)

    for i in range(T):
        f_k = freqs[i]

        valid = np.isfinite(f_k)
        if np.sum(valid) < min_partials:
            continue

        k = k_full[valid]
        f = f_k[valid]

        # fundamental estimate
        f0 = f[0]

        # Eq. (15): ck
        c_k = f - k * f0

        # Eq. (16): polynomial fit
        X = np.vstack([k**3, k, np.ones_like(k)]).T

        try:
            a, b, c = np.linalg.lstsq(X, c_k, rcond=None)[0]

            # Eq. (17): beta
            beta_n = 2 * a / (f0 + b)

            # eig. beta0
            beta = beta_n * 2**(-fret/6)

            # sanity checks
            if np.isfinite(beta) and beta > 0:
                betas[i] = beta

        except np.linalg.LinAlgError:
            continue

    return betas

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


    # statistical measures
    median_freq_deviations = np.nanmedian(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)
    mean_freq_deviations = np.nanmean(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)
    min_freq_deviations = np.nanmin(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)
    max_freq_deviations = np.nanmax(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)
    std_freq_deviations = np.nanstd(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)
    var_freq_deviations = np.nanvar(
        abs_freq_deviations / freqs_k,
        axis=0
    )  # (K-1,)

    freq_deviation_measures = np.array([
        median_freq_deviations,
        mean_freq_deviations,
        min_freq_deviations,
        max_freq_deviations,
        std_freq_deviations,
        var_freq_deviations,
    ])

    return freq_deviation_measures


def process_single_file(args):
    filepath, beta_max, plot, threshold = args

    print(f"Processing {filepath}")

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

    for note in track.valid_notes:
        betas = estimate_inharmonicity_coefficient_all_frets(
            note.partials,
            note.attributes.fret
        )
        if len(betas) == 0:
            note.invalidate(FilterReason.NO_BETAS, step="calculate_features")
            continue

        if note.features is None:
            note.features = Features()

        betas_filtered = filter_betas(betas, beta_max)
        if len(betas_filtered) == 0:
            note.invalidate(FilterReason.NO_BETAS_AFTER_FILTER, step="calculate_features")
            continue

        # assign attributed f0 to features
        note.features.f0 = note.attributes.pitch

        median_beta = np.median(betas_filtered, axis=0)

        betas_measures = np.array([
            np.mean(betas_filtered, axis=0),
            median_beta,
            np.std(betas_filtered, axis=0),
            np.var(betas_filtered, axis=0),
            np.min(betas_filtered, axis=0),
            np.max(betas_filtered, axis=0)]) # todo: skewness & curtuosis, mode

        amp_deviation_measures, amp_decay_coefficients = relative_amplitude_deviations(note.partials) # (25,)
        freq_deviation_measures = relative_freq_deviations(note.partials, median_beta)

        # valid_slopes = np.isfinite(amp_decay_coefficients)
        valid_partials = np.isfinite(amp_decay_coefficients).astype(int) # make 0 and 1

        # write into note.features
        note.features.beta = median_beta
        note.features.betas_measures = betas_measures

        note.features.valid_partials = valid_partials
        note.features.rel_partial_amplitudes = amp_deviation_measures
        note.features.rel_freq_deviations = freq_deviation_measures
        note.features.amp_decay_coefficients = amp_decay_coefficients  # was on note, not note.features
        note.features.fill_feature_vector()

        if note.features is None:
            note.invalidate(FilterReason.NO_FEATURES, step="calculate_features")


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

    track_directory = config.get('paths', 'track_directory')

    print(W, H, beta_max, threshold)
    print(track_directory)

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    args_list = [(fp, beta_max, plot, threshold) for fp in filepaths]

    # Create pool and process files
    num_processes = mp.cpu_count() - 1  # Leave one core free
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, args_list)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # load config config
    config = ConfigParser()
    config.read('config_train.ini')

    main(config)