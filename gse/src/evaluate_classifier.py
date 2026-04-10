import os
import sys
sys.path.append(os.path.abspath(''))

import argparse
import pickle
import warnings
import numpy as np
from configparser import ConfigParser
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from gse.src.feature_extraction import calculate_features
from gse.src.utils.Track_dataclass import filter_analysis


# ── Config ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

SUBSET_CONFIGS = {
    'solo': os.path.join(_DIR, 'configs', 'config_test_solo.ini'),
    'comp': os.path.join(_DIR, 'configs', 'config_test_comp.ini'),
}

STRING_NAMES = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

# Default: all filters active
FILTER_CONFIG_DEFAULT = {
    "physical_range":   True,
    "centroid_penalty": True,
    "string_occupancy": True,
}

FILTER_CONFIG_OFF = {k: False for k in FILTER_CONFIG_DEFAULT}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_fret(midi_pitch, string_index):
    open_string_midi = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    return midi_pitch - open_string_midi[string_index]


class FretboardCentroid:
    def __init__(self, window_size=5, max_hand_span=6):
        self.window_size   = window_size
        self.max_hand_span = max_hand_span
        self.recent_frets  = []

    def update(self, fret, timestamp):
        self.recent_frets.append((fret, timestamp))
        if len(self.recent_frets) > self.window_size:
            self.recent_frets.pop(0)

    def get_centroid(self, current_time):
        if not self.recent_frets:
            return None
        weights = [np.exp(-(current_time - ts) * 2.0) for _, ts in self.recent_frets]
        frets   = np.array([f for f, _ in self.recent_frets])
        return np.average(frets, weights=weights)


def get_occupied_strings(current_note, all_notes, all_probs, tolerance=0.05):
    occupied       = set()
    current_onset  = current_note.attributes.onset
    current_offset = current_note.attributes.offset
    for other_note, other_prob in zip(all_notes, all_probs):
        if other_note is current_note:
            continue
        overlaps = (other_note.attributes.onset  < current_offset - tolerance and
                    other_note.attributes.offset > current_onset  + tolerance)
        if overlaps:
            occupied.add(int(np.argmax(other_prob)))
    return occupied


# ── Plausibility filter ───────────────────────────────────────────────────────
def plausibility_filter(probabilities, notes, centroid_tracker, config=None):
    if config is None:
        config = FILTER_CONFIG_DEFAULT

    string_ranges = {
        0: (40, 60), 1: (45, 65), 2: (50, 70),
        3: (55, 75), 4: (59, 79), 5: (64, 84),
    }

    filtered_probabilities = []

    for note, prob in zip(notes, probabilities):
        midi_pitch  = note.attributes.midi_note
        timestamp   = note.attributes.onset
        masked_prob = prob.copy()

        if config["physical_range"]:
            for string in range(len(prob)):
                low, high = string_ranges[string]
                if not (low <= midi_pitch <= high):
                    masked_prob[string] = 0.0

        if config["centroid_penalty"]:
            centroid = centroid_tracker.get_centroid(timestamp)
            if centroid is not None:
                for string in range(len(prob)):
                    if masked_prob[string] == 0.0:
                        continue
                    fret     = get_fret(midi_pitch, string)
                    distance = abs(fret - centroid)
                    if distance > centroid_tracker.max_hand_span:
                        masked_prob[string] *= 0.1
                    else:
                        sigma = centroid_tracker.max_hand_span / 2
                        masked_prob[string] *= np.exp(-(distance**2) / (2 * sigma**2))

        if config["string_occupancy"]:
            occupied = get_occupied_strings(note, notes, filtered_probabilities or probabilities)
            for string in occupied:
                masked_prob[string] *= 0.05

        total       = masked_prob.sum()
        masked_prob = masked_prob / total if total > 0 else prob.copy()

        centroid_tracker.update(get_fret(midi_pitch, np.argmax(masked_prob)), timestamp)
        filtered_probabilities.append(masked_prob)

    return np.array(filtered_probabilities)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_classification(y_true, y_pred, string_labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if string_labels is None:
        string_labels = sorted(set(y_true) | set(y_pred))

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')

    per_string_precision = precision_score(y_true, y_pred, average=None, labels=string_labels)
    per_string_recall    = recall_score(y_true, y_pred, average=None, labels=string_labels)
    per_string_f1        = f1_score(y_true, y_pred, average=None, labels=string_labels)

    per_string_metrics = {}
    for i, label in enumerate(string_labels):
        mask = y_true == label
        per_string_metrics[label] = {
            'name':      STRING_NAMES[label],
            'accuracy':  accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() > 0 else None,
            'precision': per_string_precision[i],
            'recall':    per_string_recall[i],
            'f1':        per_string_f1[i],
        }

    cm = confusion_matrix(y_true, y_pred, labels=string_labels)
    return {
        'overall':          {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1},
        'per_string':       per_string_metrics,
        'confusion_matrix': cm,
        'string_labels':    string_labels,
        'string_names':     [STRING_NAMES[l] for l in string_labels],
    }


# ── Confusion matrix plots ────────────────────────────────────────────────────
def plot_confusion_matrix(results, normalize=True, title='String Classification', subset_label=None):
    """Single confusion matrix plot (standard matplotlib style)."""
    cm         = results['confusion_matrix'].copy().astype(float)
    labels     = results['string_names']
    full_title = f"{title} ({subset_label})" if subset_label else title

    if normalize:
        cm  = cm / cm.sum(axis=1, keepdims=True)
        fmt, vmax = '.2f', 1.0
    else:
        fmt, vmax = 'd', None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=vmax, ax=ax)
    ax.set_xlabel('Predicted string')
    ax.set_ylabel('True string')
    ax.set_title(full_title)
    plt.tight_layout()
    plt.show()


def plot_combined_confusion_matrices(results_solo, results_comp, normalize=True, output_path=None):
    """
    Plots Solo (top) and Comp (bottom) confusion matrices stacked vertically,
    formatted for LaTeX inclusion: 3.52 in wide, Computer Modern serif font,
    fontsize 7 pt, string labels only on both axes in horizontal orientation.
    Requires a working LaTeX installation for text.usetex=True.
    """
    # LaTeX-style rc params (restore after)
    rc = {
        'text.usetex':        True,
        'font.family':        'serif',
        # 'font.serif':         ['Computer Modern Roman'],
        'axes.labelsize':     10,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'axes.titlesize':     10,
    }

    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(
            nrows=2, ncols=1,
            figsize=(3.40, 3.40),   # width × ~1.75 for two stacked panels
        )

        for ax, results, title in zip(
            axes,
            [results_solo, results_comp],
            ['Solo', 'Comping'],
        ):
            cm = results['confusion_matrix'].copy().astype(float)
            if normalize:
                cm  = cm / cm.sum(axis=1, keepdims=True)
                fmt, vmax = '.2f', 1.0
            else:
                fmt, vmax = 'd', None

            labels = results['string_names']   # ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=vmax,
                ax=ax,
                annot_kws={'size': 9},
                cbar=False,
                linewidths=0.5,
                linecolor='lightgray',
            )

            # Horizontal tick labels on both axes
            ax.set_xticklabels(labels, rotation=0, ha='center')
            ax.set_yticklabels(labels, rotation=0, va='center')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(title)

        plt.tight_layout(pad=0.5)

        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"[Plot] Saved combined confusion matrix to: {output_path}")
        else:
            plt.show()


# ── Permutation importance ────────────────────────────────────────────────────
def get_feature_groups(sample_fv):
    segments = {
        "f0":                     1,
        "betas_measures":         sample_fv.betas_measures.size,
        "valid_partials":         sample_fv.valid_partials.size,
        "spectral_centroid":      sample_fv.spectral_centroid.size,
        "rel_partial_amplitudes": sample_fv.rel_partial_amplitudes.size,
        "amp_decay_coefficients": sample_fv.amp_decay_coefficients.size,
        "rel_freq_deviations":    sample_fv.rel_freq_deviations.size,
    }
    groups, offset = {}, 0
    for name, size in segments.items():
        groups[name] = list(range(offset, offset + size))
        offset += size
    return groups


def get_stat_groups(groups, measure_segments_with_k,
                    stat_names=["median","mean","min","max","std","var","skewness","kurtosis","mode"]):
    stat_groups = {name: [] for name in stat_names}
    for seg_name, k in measure_segments_with_k.items():
        stat_index_blocks = np.array(groups[seg_name]).reshape(len(stat_names), k)
        for i, stat_name in enumerate(stat_names):
            stat_groups[stat_name].extend(stat_index_blocks[i].tolist())
    return stat_groups


def grouped_permutation_importance(model, X, y, groups, n_repeats=5, random_state=42):
    rng      = np.random.RandomState(random_state)
    baseline = accuracy_score(y, model.predict(X))
    results  = {}
    for group_name, col_indices in groups.items():
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            shuffle_idx = rng.permutation(len(y))
            X_perm[:, col_indices] = X_perm[shuffle_idx][:, col_indices]
            scores.append(baseline - accuracy_score(y, model.predict(X_perm)))
        results[group_name] = {"mean": np.mean(scores), "std": np.std(scores)}
    return results


def run_permutation(SVM, all_features, all_labels, all_valid_notes, audio_types):
    print("\n[Permutation] Running grouped permutation importance...")
    FX_test    = np.stack(all_features)
    labels_test = np.array(all_labels)

    sample_fv      = all_valid_notes[0].features[audio_types[0]]
    feature_groups = get_feature_groups(sample_fv)
    measure_segs   = {
        "rel_partial_amplitudes": sample_fv.rel_partial_amplitudes.shape[1],
        "rel_freq_deviations":    sample_fv.rel_freq_deviations.shape[1],
    }
    stat_groups = get_stat_groups(feature_groups, measure_segs)

    seg_imp  = grouped_permutation_importance(SVM, FX_test, labels_test, feature_groups)
    stat_imp = grouped_permutation_importance(SVM, FX_test, labels_test, stat_groups)

    print("\n── Segment importance ──")
    for name, res in sorted(seg_imp.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {name:30s}  {res['mean']:.4f} ± {res['std']:.4f}")

    print("\n── Stat importance ──")
    for name, res in sorted(stat_imp.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {name:30s}  {res['mean']:.4f} ± {res['std']:.4f}")


# ── Core evaluation run for one subset ───────────────────────────────────────
def run_subset(subset, SVM, filter_config):
    config_test = ConfigParser()
    config_test.read(SUBSET_CONFIGS[subset])

    # --- Step 2: Calculate features ---
    # calculate_features.main(config_test)

    track_directory = config_test.get('paths', 'track_directory')
    audio_types     = [a.strip() for a in config_test.get('paths', 'audio_types').split(',')]

    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values",
        category=UserWarning,
        module="sklearn.impute",
    )

    filepaths = [
        os.path.join(track_directory, fn)
        for fn in os.listdir(track_directory)
        if fn.endswith(".pkl") and os.path.isfile(os.path.join(track_directory, fn))
    ]

    all_predictions, all_features, all_labels = [], [], []
    all_notes, all_valid_notes = [], []

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] [{subset}] Classification: {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)

        centroid_tracker = FretboardCentroid()

        for note in track.notes:
            all_notes.append(note)

        for audio_type in audio_types:
            track_fvs, track_notes = [], []
            for note in track.valid_notes:
                if audio_type not in note.features:
                    continue
                fv = note.features[audio_type].feature_vector
                if fv is None:
                    continue
                track_fvs.append(fv)
                track_notes.append(note)
                all_features.append(fv)
                all_labels.append(note.attributes.string_index)
                all_valid_notes.append(note)

            if not track_fvs:
                continue

            track_preds = SVM.predict_proba(np.stack(track_fvs))
            track_preds = plausibility_filter(track_preds, track_notes, centroid_tracker, config=filter_config)
            all_predictions.extend(track_preds)

    filter_analysis(all_notes)

    string_labels = [0, 1, 2, 3, 4, 5]
    results       = evaluate_classification(
        all_labels,
        np.argmax(all_predictions, axis=1),
        string_labels=string_labels,
    )

    print(results['overall'])
    print(results['per_string'])

    return results, all_features, all_labels, all_valid_notes, audio_types


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SVM Evaluation – GuitarSet")
    parser.add_argument(
        "--subset",
        choices=["solo", "comp", "both"],
        default=None,
        help="Test subset: 'solo', 'comp', or 'both' (also generates combined LaTeX plot).",
    )
    parser.add_argument(
        "--permutation",
        action="store_true",
        default=None,
        help="Run grouped permutation importance after evaluation.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable the plausibility filter entirely.",
    )
    args = parser.parse_args()

    # Interactive: subset
    if args.subset is None:
        print("\nSelect test subset:")
        print("  [1] solo")
        print("  [2] comp")
        print("  [3] both  (also generates combined LaTeX confusion matrix plot)")
        choice = input("\nEnter 1, 2, or 3: ").strip()
        args.subset = {"1": "solo", "2": "comp", "3": "both"}.get(choice, "solo")

    # Interactive: permutation
    if args.permutation is None:
        choice = input("\nRun permutation importance? [y/N]: ").strip().lower()
        args.permutation = choice == "y"

    # Interactive: plausibility filter
    if not args.no_filter:
        choice = input("\nEnable plausibility filter? [Y/n]: ").strip().lower()
        filter_config = FILTER_CONFIG_OFF if choice == "n" else FILTER_CONFIG_DEFAULT
    else:
        filter_config = FILTER_CONFIG_OFF

    active = [k for k, v in filter_config.items() if v]
    print(f"\n[Filter] Active: {active if active else 'none'}")

    SVM = joblib.load("svm_gridsearch_30pct.joblib")

    subsets_to_run = ["solo", "comp"] if args.subset == "both" else [args.subset]
    all_results    = {}

    for subset in subsets_to_run:
        print(f"\n{'=' * 50}\n  Running subset: {subset}\n{'=' * 50}")
        results, features, labels, valid_notes, audio_types = run_subset(subset, SVM, filter_config)
        all_results[subset] = results

        # 👉 F1 pro Subset
        f1 = results['overall']['f1']
        print(f"[F1] {subset}: {f1:.4f}")

        plot_confusion_matrix(results, subset_label=subset)

        if args.permutation:
            run_permutation(SVM, features, labels, valid_notes, audio_types)

    # 👉 Mean F1 über Subsets
    if len(all_results) > 1:
        f1_scores = [res['overall']['f1'] for res in all_results.values()]
        mean_f1 = np.mean(f1_scores)

        print("\n" + "=" * 50)
        print(f"[F1] Mean over subsets: {mean_f1:.4f}")
        print("=" * 50)

    # Combined LaTeX plot when both subsets were evaluated
    if args.subset == "both" and "solo" in all_results and "comp" in all_results:
        print("\n[Plot] Generating combined LaTeX confusion matrix...")
        plot_combined_confusion_matrices(
            all_results["solo"],
            all_results["comp"],
            output_path="confusion_matrix_combined.pdf",
        )

    print("\n\nSuccess! Classification done.")


if __name__ == "__main__":
    main()