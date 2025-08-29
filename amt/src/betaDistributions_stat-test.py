import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, shapiro, bartlett, levene, f_oneway


def load_betas_from_json(filename):
    """Lädt die Beta-Werte aus einer JSON-Datei."""
    with open(filename, 'r') as f:
        return json.load(f)


def check_variance_homogeneity(saiten_werte, test_type="levene"):
    """Testet die Homogenität der Varianzen (Bartlett oder Levene)."""
    if test_type == "bartlett":
        stat, p_value = bartlett(*saiten_werte)
        test_name = "Bartlett-Test"
    elif test_type == "levene":
        stat, p_value = levene(*saiten_werte)
        test_name = "Levene-Test"
    else:
        raise ValueError("Invalid test_type. Choose either 'bartlett' or 'levene'.")

    return test_name, stat, p_value


def perform_welch_anova(saiten_werte):
    """Führt eine Welch-ANOVA durch."""
    f_stat, p_value_anova = f_oneway(*saiten_werte)
    return f_stat, p_value_anova


import numpy as np
import matplotlib.pyplot as plt

def plot_beta_distributions(betas):
    """Plots histograms with relative frequencies for the given beta values and prints case counts per string."""
    colors = ['skyblue', 'red', 'green', 'orange', 'purple', 'brown']
    saiten_werte = []

    for saite_index in range(1, 7):
        saite_name = f"Saite_{saite_index}"
        werte = np.array([w for w in betas.get(saite_name, []) if 0 <= w <= 0.001])  # Filter values between 0 and 0.001

        anzahl_faelle = len(werte)
        print(f"{saite_name}: {anzahl_faelle} Fälle")  # Anzahl der Fälle ausgeben

        if werte.size == 0:
            print(f"Keine gültigen Werte für {saite_name}.")
            continue

        saiten_werte.append(werte)
        color = colors[saite_index - 1]

        # Calculate histogram with relative frequency
        hist_values, bins = np.histogram(werte, bins=100, density=False)  # density=False for absolute frequencies
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = np.diff(bins)

        # Calculate relative frequency
        hist_values_rel = hist_values / np.sum(hist_values)  # Normalize frequencies to sum to 1

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist_values_rel, width=bin_width[0], alpha=0.6, color=color, edgecolor='black',
                label="Relative Häufigkeit")

        plt.title(f'Relative Häufigkeit: {saite_name}')
        plt.xlabel('Beta-Werte')
        plt.ylabel('Relative Häufigkeit')
        plt.legend()
        plt.xlim(0, 0.001)  # Set x-axis limits to the range of beta values
        plt.show()

    return saiten_werte

def main():
    """Hauptfunktion zum Laden der Daten, Testen und Plotten."""
    filename = '../content/Betas/betas.json'
    betas = load_betas_from_json(filename)
    saiten_werte = plot_beta_distributions(betas)

    if len(saiten_werte) > 1:
        test_name, var_stat, var_p = check_variance_homogeneity(saiten_werte)
        print(f"{test_name}: Statistik = {var_stat:.3f}, p-Wert = {var_p:.3f}")

        f_stat, p_value_anova = perform_welch_anova(saiten_werte)
        print(f"Welch-ANOVA - F-Statistik: {f_stat:.3f}, p-Wert: {p_value_anova:.3f}")


if __name__ == "__main__":
    main()
