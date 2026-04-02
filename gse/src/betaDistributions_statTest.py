import configparser
import os
import sys


sys.path.append(os.path.abspath(''))
import multiprocessing as mp
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, shapiro, bartlett, levene, f_oneway
from typing import Dict, List, Tuple
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

from gse.src import calculate_features


def check_variance_homogeneity(string_values: List[np.ndarray],
                               test_type: str = "levene") -> Tuple[str, float, float]:
    if test_type == "bartlett":
        stat, p_value = bartlett(*string_values)
        test_name = "Bartlett's Test"
    elif test_type == "levene":
        stat, p_value = levene(*string_values)
        test_name = "Levene's Test"
    else:
        raise ValueError("Invalid test_type. Choose either 'bartlett' or 'levene'.")

    return test_name, stat, p_value


def perform_welch_anova(string_values: List[np.ndarray]) -> Tuple[float, float]:
    f_stat, p_value_anova = f_oneway(*string_values)
    return f_stat, p_value_anova

def plot_beta_distributions(betas: Dict[int, List[float]]) -> List[np.ndarray]:
    colors = ['skyblue', 'red', 'green', 'orange', 'purple', 'brown']
    string_labels = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
    string_values = []

    # --- Pass 1: collect all valid values to define global bins ---
    all_values = []
    for string_index in sorted(betas.keys()):
        values_raw = np.array(betas.get(string_index, []), dtype=float)
        values = values_raw[np.isfinite(values_raw)]
        all_values.append(values)

    global_min = min(v.min() for v in all_values if v.size > 0)
    global_max = 2e-4
    global_bins = np.linspace(global_min, global_max, 201)  # 200 bins, uniform width
    bin_width = global_bins[1] - global_bins[0]
    bin_centers = (global_bins[:-1] + global_bins[1:]) / 2

    # --- Pass 2: individual plots + collect string_values ---
    for string_index, values in zip(sorted(betas.keys()), all_values):
        string_name = f"String {string_index} ({string_labels[string_index]})"

        dropped = len(np.array(betas.get(string_index, []), dtype=float)) - len(values)
        if dropped > 0:
            print(f"{string_name}: dropped {dropped} non-finite values")
        print(f"{string_name}: {len(values)} cases")

        if values.size == 0:
            print(f"No valid values for {string_name}.")
            continue

        string_values.append(values)
        color = colors[string_index % len(colors)]

        hist_values, _ = np.histogram(values, bins=global_bins, density=False)
        hist_values_rel = hist_values / np.sum(hist_values)

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist_values_rel, width=bin_width, alpha=0.6,
                color=color, edgecolor='none', label="Relative Frequency")
        plt.title(f'Beta Distribution: {string_name} 3rdtaylor')
        plt.xlabel('Beta')
        plt.ylabel('Relative Frequency')
        plt.xlim(global_min, global_max)
        plt.legend()
        plt.tight_layout()

    # --- Overlay plot ---
    DPI = 800  # change to taste
    # overlay plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    for i, (string_index, values) in enumerate(zip(sorted(betas.keys()), string_values)):
        color = colors[string_index % len(colors)]
        label = f"String {string_index} ({string_labels[string_index]})"

        hist_values, _ = np.histogram(values, bins=global_bins, density=False)
        hist_values_norm = hist_values / hist_values.max()  # peak-normalised → all reach 1.0

        ax.bar(bin_centers, hist_values_norm, width=bin_width,
               alpha=0.4, color=color, edgecolor='none', label=label)
        ax.plot(bin_centers, hist_values_norm, color=color, linewidth=1.2, alpha=0.85)

    ax.set_title('Beta – Normierte Wahrscheinlichkeitsdichtefunktionen')
    ax.set_ylabel('Normalisierte Wahrscheinlichkeit')
    ax.set_xlabel('Beta')
    ax.set_xlim(global_min, global_max)
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    return string_values


def main(config):
    track_directory = config.get('paths', 'track_directory')
    audio_types_raw = config.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]
    beta0_min = config.getfloat('params', 'beta0_min')
    beta0_max = config.getfloat('params', 'beta0_max')
    beta_min = beta0_min  # a fret only scales beta upwards
    beta_max = beta0_max * 2 ** (20 / 6)  # 20th fret as large boundary

    # --- Step 2: Calculate features ---
    calculate_features.main(config)

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if filename.endswith(".pkl")
        if os.path.isfile(os.path.join(track_directory, filename))
        # and "solo" in filename
    ]
    # Initialize dictionary to collect betas by string index
    betas_by_string = {i: [] for i in range(6)}


    all_notes = []
    all_betas = []

    for audio_type in audio_types:
        for filepath in filepaths:
            with open(filepath, "rb") as f:
                try:
                    track = pickle.load(f)
                except EOFError:
                    continue

            all_notes.extend(track.valid_notes)
            for note in track.valid_notes:
                if audio_type not in note.features:
                    continue
                if note.features[audio_type].beta is None:
                    continue
                string_index = note.attributes.string_index
                if note.origin == "single_note":
                    string_index -= 1

                betas_by_string[string_index].append(note.features[audio_type].beta0(note.attributes.fret))
                all_betas.append(note.features[audio_type].beta0(note.attributes.fret))

    cleaned_betas_by_string = plot_beta_distributions(betas_by_string)

    betas_by_string_ndarray = np.array(
        [np.array(cleaned_betas_by_string[i]) for i in range(6)],
        dtype=object
    )

    p = Path(track_directory)

    last_three = p.parts[-3:]
    betas_savename = "betas_" + "_".join(last_three) + ".npy"
    np.save(betas_savename, betas_by_string_ndarray)


    print(f"Number of notes total: {len(all_notes)}\n")
    print(f"Ratio of notes with beta: {len(all_betas)/len(all_notes)}\n")



    # Perform statistical tests if we have multiple strings with noteData
    if len(cleaned_betas_by_string) > 1:
        # Test for variance homogeneity
        test_name, var_stat, var_p = check_variance_homogeneity(cleaned_betas_by_string)
        print(f"{test_name}: Statistic = {var_stat:.3f}, p-value = {var_p:.3f}")

        # Perform ANOVA -> vielleich Welch?
        f_stat, p_value_anova = perform_welch_anova(cleaned_betas_by_string)
        print(f"ANOVA - F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.3f}")


if __name__ == "__main__":
    # test on the following subsets:
    config = configparser.ConfigParser()
    config.read('configs/config_train_dev_GuitarSet.ini')

    main(config)
    # main(subset_comp)
    # main(subset_GOAT)
    # main(subset_single_note_IDMT)