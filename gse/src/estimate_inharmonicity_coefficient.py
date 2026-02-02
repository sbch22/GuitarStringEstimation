import os
import sys


sys.path.append(os.path.abspath(''))
import multiprocessing as mp
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
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


def estimate_inharmonicity_coefficient_iterative(
    partials,
    beta_max,
    plot,
    n_iter=200,
    tol=1e-9,
    amp_threshold_db = -50
):
    freqs = partials.frequencies  # (T, K)
    amps = partials.amplitudes
    T, K = freqs.shape

    betas = np.full(T, np.nan)
    k_full = np.arange(1, K + 1)

    beta = 5e-5 # start harmonic

    for i in range(T):
        f_k = freqs[i]
        a_k = amps[i]

        valid = (~np.isnan(f_k)) & (a_k > amp_threshold_db) & (k_full < 25)

        if np.sum(valid) < 15:
            continue

        k = k_full[valid]
        f = f_k[valid]
        a = a_k[valid]

        f0 = f[0]

        # initial f0 estimate
        if np.isnan(f0):
            break


        # iterative beta-Schätzung
        for _ in range(n_iter):
            f_expected = k * f0 * np.sqrt(1 + beta * k ** 2)
            delta_f = f - f_expected

            valid2 = np.abs(delta_f) <= 100

            # konsistent auf aktuelle k,f,a anwenden
            k2 = k[valid2]
            f2 = f[valid2]
            a2 = a[valid2]
            delta_f2 = delta_f[valid2]

            # genug Punkte behalten?
            if len(k2) < 5:
                break

            X = np.vstack([k2 ** 3, k2, np.ones_like(k2)]).T

            weights = a2 - np.min(a2)
            if np.max(weights) > 0:
                weights /= np.max(weights)
            else:
                weights = np.ones_like(weights)

            W = np.sqrt(weights)

            Xw = X * W[:, None]
            delta_fw = delta_f2 * W

            tp_a, tp_b, tp_c = np.linalg.lstsq(Xw, delta_fw, rcond=None)[0]

            beta_new = 2 * tp_a / (f0 + tp_b)
            if np.abs(beta_new - beta) < tol:
                beta = beta_new
                break
            beta = beta_new

        betas[i] = beta

        if plot:
            k_fit = np.linspace(k.min(), k.max(), 500)
            delta_fit = tp_a * k_fit**3 + tp_b * k_fit + tp_c

            plt.figure(figsize=(6, 4))
            plt.scatter(k, delta_f, c=weights, cmap="viridis")
            plt.colorbar(label="weight")
            plt.plot(k_fit, delta_fit, label="fit")
            plt.xlabel("partial index k")
            plt.ylabel("Δfₖ (Hz)")
            plt.title(f"t={i}, β={beta:.2e}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    betas = filter_betas(betas, beta_max)
    return betas


def process_single_file(args):
    filepath, beta_max = args

    try:
        with open(filepath, "rb") as f:
            track = pickle.load(f)

        # not all notes are created equal. Some do not have partials ...
        for note in track.notes:
            if note.match is not True:
                continue
            if note.origin != 'model':
                continue
            if note.partials is None:
                continue
            if note.partials.frequencies is None:
                continue

            betas = estimate_inharmonicity_coefficient_iterative(note.partials, beta_max, plot=False)

            if note.features is None:
                note.features = Features()

            note.features.betas = betas

        track.save(filepath)
        print(f"pickled beta-calculated note object into {filepath}.")

        filename = os.path.basename(filepath)
        return f"Success: {filename}"

    except Exception as e:
        return f"Error processing {filepath}: {str(e)}"


def main():
    track_directory = '../noteData/GuitarSet/train/dev/'

    # Parameters
    beta_max = 2e-4

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    args_list = [(fp, beta_max) for fp in filepaths]

    # Create pool and process files
    num_processes = mp.cpu_count() - 1  # Leave one core free
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, args_list)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()