import os
import sys


sys.path.append(os.path.abspath(''))
import multiprocessing as mp
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def filter_betas(betas):
    valid = ~np.isnan(betas)
    valid_betas = betas[valid]

    # Todo: IQR filter

    return valid_betas

def estimate_inharmonicity_coefficient(partials, beta_max, plot):
    freqs = partials.frequencies  # (T, K)
    T, K = freqs.shape

    betas = np.full(T, np.nan)

    # fixed partial indices (1-based)
    k_full = np.arange(1, K + 1)

    for i in range(T):
        f_k = freqs[i]

        valid = ~np.isnan(f_k)

        # need enough valid partials
        if np.sum(valid) < 5:
            continue

        k = k_full[valid]
        f = f_k[valid]

        f0 = f[0]  # assumes fundamental exists

        delta_f = f - k * f0

        X = np.vstack([k**3, k, np.ones_like(k)]).T
        coeffs, _, _, _ = np.linalg.lstsq(X, delta_f, rcond=None)
        a, b, c = coeffs

        beta = np.clip(2 * a / f0, 0.0, beta_max)
        betas[i] = beta

        if plot:
            k_fit = np.linspace(k.min(), k.max(), 500)
            delta_fit = a * k_fit**3 + b * k_fit + c

            plt.figure(figsize=(6, 4))
            plt.scatter(k, delta_f)
            plt.plot(k_fit, delta_fit)
            plt.xlabel("partial index k")
            plt.ylabel("Δfₖ (Hz)")
            plt.title(f"t={i}, β={beta:.2e}")
            plt.tight_layout()
            plt.show()

    # filter betas
    betas = filter_betas(betas)
    return betas

def main():
    track_directory = '../noteData/'

    # Parameters
    beta_max = 1e-3

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    # TODO: implement parallelization via multiprocessing
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            track = pickle.load(f)

        # not all notes are created equal. Some do not have partials ...
        for note in track.notes:
            if note.match is not True:
                continue
            if note.partials is None:
                continue
            if note.partials.frequencies is None:
                continue

            betas = estimate_inharmonicity_coefficient(note.partials, beta_max, plot=False)

            note.attributes.betas = betas

        # Save track
        track.save(filepath)


if __name__ == "__main__":
    main()