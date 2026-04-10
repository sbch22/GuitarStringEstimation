import os
import sys
import pickle
import configparser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bartlett, levene, f_oneway

sys.path.append(os.path.abspath(""))

# ── Constants ────────────────────────────────────────────────────────────────
STRING_LABELS = ["E2", "A2", "D3", "G3", "B3", "E4"]
STRING_COLORS = ["skyblue", "red", "green", "orange", "purple", "brown"]
NUM_STRINGS = 6
BETA_UPPER_LIMIT = 2e-4
NUM_BINS = 200
PLOT_DPI = 800
SAVE_DIR = Path("../../data/GuitarSet/noteData")


# ── Statistical helpers ──────────────────────────────────────────────────────
def check_variance_homogeneity(
    groups: List[np.ndarray], test: str = "levene"
) -> Tuple[str, float, float]:
    """Run Bartlett's or Levene's test for homogeneity of variances."""
    if test == "bartlett":
        stat, p = bartlett(*groups)
        return "Bartlett's Test", stat, p
    if test == "levene":
        stat, p = levene(*groups)
        return "Levene's Test", stat, p
    raise ValueError(f"Unknown test '{test}'. Use 'bartlett' or 'levene'.")


def perform_welch_anova(groups: List[np.ndarray]) -> Tuple[float, float]:
    """One-way ANOVA (Welch variant via scipy)."""
    return f_oneway(*groups)


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_beta_distributions(
    betas: Dict[int, List[float]],
) -> List[np.ndarray]:
    """Plot per-string and overlay histograms; return cleaned arrays."""

    # Clean values per string
    cleaned: Dict[int, np.ndarray] = {}
    for idx in sorted(betas):
        raw = np.asarray(betas[idx], dtype=float)
        finite = raw[np.isfinite(raw)]
        dropped = len(raw) - len(finite)
        name = f"String {idx} ({STRING_LABELS[idx]})"
        if dropped:
            print(f"{name}: dropped {dropped} non-finite values")
        print(f"{name}: {len(finite)} cases")
        if finite.size:
            cleaned[idx] = finite

    if not cleaned:
        print("No valid beta values to plot.")
        return []

    global_min = min(v.min() for v in cleaned.values())
    bins = np.linspace(global_min, BETA_UPPER_LIMIT, NUM_BINS + 1)
    bin_width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    # Per-string histograms
    for idx, values in cleaned.items():
        color = STRING_COLORS[idx % len(STRING_COLORS)]
        label = f"String {idx} ({STRING_LABELS[idx]})"

        hist, _ = np.histogram(values, bins=bins)
        rel_freq = hist / hist.sum()

        plt.figure(figsize=(8, 5))
        plt.bar(centers, rel_freq, width=bin_width, alpha=0.6,
                color=color, edgecolor="none", label="Relative Frequency")
        plt.title(f"Beta Distribution: {label}")
        plt.xlabel("Beta")
        plt.ylabel("Relative Frequency")
        plt.xlim(global_min, BETA_UPPER_LIMIT)
        plt.legend()
        plt.tight_layout()

    # Overlay plot (peak-normalised)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=PLOT_DPI)
    for idx, values in cleaned.items():
        color = STRING_COLORS[idx % len(STRING_COLORS)]
        label = f"String {idx} ({STRING_LABELS[idx]})"

        hist, _ = np.histogram(values, bins=bins)
        normed = hist / hist.max()

        ax.bar(centers, normed, width=bin_width, alpha=0.4,
               color=color, edgecolor="none", label=label)
        ax.plot(centers, normed, color=color, linewidth=1.2, alpha=0.85)

    ax.set_title("Beta – Normierte Wahrscheinlichkeitsdichtefunktionen")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Normalisierte Wahrscheinlichkeit")
    ax.set_xlim(global_min, BETA_UPPER_LIMIT)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

    return [cleaned[i] for i in sorted(cleaned)]


# ── Beta collection ──────────────────────────────────────────────────────────
def collect_betas(
    track_directory: str,
    audio_types: List[str],
    note_filter: str = "solo",
) -> Tuple[Dict[int, list], list, list]:
    """Load pickled tracks and extract betas grouped by string index."""

    filepaths = [
        os.path.join(track_directory, f)
        for f in os.listdir(track_directory)
        if f.endswith(".pkl")
        and os.path.isfile(os.path.join(track_directory, f))
        and note_filter in f
    ]

    betas_by_string: Dict[int, list] = {i: [] for i in range(NUM_STRINGS)}
    all_notes: list = []
    all_betas: list = []

    for audio_type in audio_types:
        for filepath in filepaths:
            with open(filepath, "rb") as fh:
                try:
                    track = pickle.load(fh)
                except EOFError:
                    continue

            all_notes.extend(track.valid_notes)

            for note in track.valid_notes:
                if audio_type not in note.features:
                    continue
                if note.features[audio_type].beta is None:
                    continue

                idx = note.attributes.string_index
                if note.origin == "single_note":
                    idx -= 1

                beta = note.features[audio_type].beta0(note.attributes.fret)
                betas_by_string[idx].append(beta)
                all_betas.append(beta)

    return betas_by_string, all_notes, all_betas


# ── Save helper ──────────────────────────────────────────────────────────────
def build_save_name(config_path: str) -> str:
    """Derive a descriptive filename from the config path.

    Example config path: 'configs/config_train_GuitarSet.ini'
    Produces:            'betas_config_train_GuitarSet.npy'
    """
    stem = Path(config_path).stem          # e.g. "config_train_GuitarSet"
    return f"betas_{stem}.npy"


def save_betas(
    cleaned: List[np.ndarray],
    config_path: str,
) -> Path:
    """Save per-string beta arrays to SAVE_DIR with config indicator."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    arr = np.array(
        [np.asarray(cleaned[i]) for i in range(len(cleaned))],
        dtype=object,
    )
    out = SAVE_DIR / build_save_name(config_path)
    np.save(out, arr)
    print(f"Saved betas → {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main(config: configparser.ConfigParser, config_path: str) -> None:
    track_directory = config.get("paths", "track_directory")
    audio_types = [a.strip() for a in config.get("paths", "audio_types").split(",")]

    betas_by_string, all_notes, all_betas = collect_betas(
        track_directory, audio_types, note_filter=""
    )

    cleaned = plot_beta_distributions(betas_by_string)
    save_betas(cleaned, config_path)

    print(f"\nTotal notes:          {len(all_notes)}")
    print(f"Notes with beta:      {len(all_betas)}")
    if all_notes:
        print(f"Beta coverage:        {len(all_betas) / len(all_notes):.2%}")

    # Statistical tests
    if len(cleaned) > 1:
        name, stat, p = check_variance_homogeneity(cleaned)
        print(f"\n{name}: statistic = {stat:.3f}, p = {p:.3f}")

        f_stat, p_anova = perform_welch_anova(cleaned)
        print(f"ANOVA:  F = {f_stat:.3f}, p = {p_anova:.3f}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config_train_GuitarSet.ini"

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    main(config, CONFIG_PATH)