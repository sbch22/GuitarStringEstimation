import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, shapiro, bartlett, levene, f_oneway
from typing import Dict, List, Tuple


def load_betas_from_json(filename: str) -> Dict[str, List[float]]:
    """
    Load beta values from a JSON file.

    Args:
        filename: Path to the JSON file containing beta values

    Returns:
        Dictionary containing beta values organized by string
    """
    with open(filename, 'r') as file:
        return json.load(file)


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


def plot_beta_distributions(betas: Dict[str, List[float]]) -> List[np.ndarray]:
    """
    Plot histograms of beta values for each guitar string with relative frequencies.

    This function:
    1. Filters beta values to the range [0, 0.001]
    2. Prints the number of valid cases per string
    3. Creates histograms showing the relative frequency distribution

    Args:
        betas: Dictionary containing beta values for each string

    Returns:
        List of arrays containing filtered beta values for each string
    """
    colors = ['skyblue', 'red', 'green', 'orange', 'purple', 'brown']
    string_values = []

    # Process each string (1 to 6)
    for string_index in range(1, 7):
        string_name = f"Saite_{string_index}"

        # Filter values to the valid range [0, 0.001]
        values = np.array([w for w in betas.get(string_name, []) if 0 <= w <= 0.001])

        # Print case count for this string
        case_count = len(values)
        print(f"{string_name}: {case_count} cases")

        # Skip if no valid values
        if values.size == 0:
            print(f"No valid values for {string_name}.")
            continue

        string_values.append(values)
        color = colors[string_index - 1]

        # Calculate histogram with absolute frequencies
        hist_values, bins = np.histogram(values, bins=100, density=False)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = np.diff(bins)

        # Convert to relative frequencies (sum to 1)
        hist_values_rel = hist_values / np.sum(hist_values)

        # Create histogram plot
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist_values_rel, width=bin_width[0], alpha=0.6,
                color=color, edgecolor='black', label="Relative Frequency")

        plt.title(f'Relative Frequency: {string_name}')
        plt.xlabel('Beta Values')
        plt.ylabel('Relative Frequency')
        plt.legend()
        plt.xlim(0, 0.001)  # Set x-axis limits to focus on relevant range
        plt.show()

    return string_values


def main():
    # Load beta values from file
    filename = '../content/Betas/betas.json'
    betas = load_betas_from_json(filename)

    # Plot distributions and get filtered values
    string_values = plot_beta_distributions(betas)

    # Perform statistical tests if we have multiple strings with data
    if len(string_values) > 1:
        # Test for variance homogeneity
        test_name, var_stat, var_p = check_variance_homogeneity(string_values)
        print(f"{test_name}: Statistic = {var_stat:.3f}, p-value = {var_p:.3f}")

        # Perform Welch's ANOVA
        f_stat, p_value_anova = perform_welch_anova(string_values)
        print(f"Welch's ANOVA - F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.3f}")


if __name__ == "__main__":
    main()