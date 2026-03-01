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


def check_variance_homogeneity(string_values: List[np.ndarray],
                               test_type: str = "levene") -> Tuple[str, float, float]:
    """
    Test for homogeneity of variances using either Bartlett's or Levene's test.

    Args:
        string_values: List of arrays containing beta values for each string
        test_type: Type of test to perform ("bartlett" or "levene")

    Returns:
        Tuple containing:
        - Name of the test performed
        - Test statistic
        - p-value

    Raises:
        ValueError: If an invalid test type is specified
    """
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
    """
    Perform Welch's ANOVA test on beta values from different strings.

    Welch's ANOVA is used when the assumption of equal variances is violated.

    Args:
        string_values: List of arrays containing beta values for each string

    Returns:
        Tuple containing:
        - F-statistic
        - p-value
    """
    f_stat, p_value_anova = f_oneway(*string_values)
    return f_stat, p_value_anova


def plot_beta_distributions(betas: Dict[int, List[float]], beta_max) -> List[np.ndarray]:
    """
    Plot histograms of beta values for each guitar string with relative frequencies.

    Args:
        betas: Dictionary containing beta values for each string (keys are integers 0-5)

    Returns:
        List of arrays containing filtered beta values for each string
    """
    colors = ['skyblue', 'red', 'green', 'orange', 'purple', 'brown']
    string_values = []

    # Create histogram plot
    # plt.figure(figsize=(8, 5))

    # Process each string present in the dictionary
    for string_index in sorted(betas.keys()):
        string_name = f"String {string_index}"

        # Filter values to the valid range [0, 0.001]
        values = np.array([w for w in betas.get(string_index, []) if 0 <= w <= beta_max])

        # Print case count for this string
        case_count = len(values)
        print(f"{string_name}: {case_count} cases")

        # Skip if no valid values
        if values.size == 0:
            print(f"No valid values for {string_name}.")
            continue

        string_values.append(values)
        color = colors[string_index % len(colors)]  # Handle index safely

        # Calculate histogram with absolute frequencies
        hist_values, bins = np.histogram(values, bins=200, density=False)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = np.diff(bins)

        # Convert to relative frequencies (sum to 1)
        hist_values_rel = hist_values / np.sum(hist_values)

        # # Create histogram plot
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist_values_rel, width=bin_width[0], alpha=0.6,
                color=color, edgecolor='black', label="Relative Frequency")

        plt.title(f'Relative Frequency: {string_name}')
        plt.xlabel('Beta Values')
        plt.ylabel('Relative Frequency')
        plt.legend()
        plt.xlim(0, 2e-4)
    plt.show()

    return string_values


def main():
    # read config
    config = ConfigParser()
    config.read('config.ini')
    beta_max = config.getfloat('train', 'beta_max')
    track_directory = config.get('paths', 'track_directory')
    print(f"Beta Max: {beta_max}")

    print(track_directory)

    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    # Initialize dictionary to collect betas by string index
    betas_by_string = {i: [] for i in range(6)}

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            try:
                track = pickle.load(f)
            except EOFError:
                continue

        for note in track.notes:
            if note.valid is not True:
                continue
            if note.features is None:
                continue
            if note.features.betas is None:
                continue

            string_index = note.attributes.string_index
            if note.origin == "single_note":
                string_index += -1
            betas_by_string[string_index].extend(note.features.betas)


    cleaned_betas_by_string = plot_beta_distributions(betas_by_string, beta_max)


    # 👉 In ndarray umwandeln (object dtype, weil unterschiedlich lang)
    betas_by_string_ndarray = np.array(
        [np.array(cleaned_betas_by_string[i]) for i in range(6)],
        dtype=object
    )
    # save ndarray in source folder as betas_dev.npy

    p = Path(track_directory)

    last_three = p.parts[-3:]
    betas_savename = "betas_" + "_".join(last_three) + ".npy"
    np.save(betas_savename, betas_by_string_ndarray)



    # Perform statistical tests if we have multiple strings with noteData
    if len(cleaned_betas_by_string) > 1:
        # Test for variance homogeneity
        test_name, var_stat, var_p = check_variance_homogeneity(cleaned_betas_by_string)
        print(f"{test_name}: Statistic = {var_stat:.3f}, p-value = {var_p:.3f}")

        # Perform ANOVA -> vielleich Welch?
        f_stat, p_value_anova = perform_welch_anova(cleaned_betas_by_string)
        print(f"ANOVA - F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.3f}")


if __name__ == "__main__":
    main()