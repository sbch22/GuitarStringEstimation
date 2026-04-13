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

FILTER_CONFIG_DEFAULT = {
    "physical_range":   True,
    "centroid_penalty": True,
    "string_occupancy": True,
}

FILTER_CONFIG_OFF = {k: False for k in FILTER_CONFIG_DEFAULT}

# Incremental filter stages for filter-importance experiment
FILTER_STAGES = [
    ("no filter",              {"physical_range": False, "centroid_penalty": False, "string_occupancy": False}),
    ("only physical_range",    {"physical_range": True,  "centroid_penalty": False, "string_occupancy": False}),
    ("only centroid_penalty",  {"physical_range": False, "centroid_penalty": True,  "string_occupancy": False}),
    ("only string_occupancy",  {"physical_range": False, "centroid_penalty": False, "string_occupancy": True}),
    ("all filters",            {"physical_range": True,  "centroid_penalty": True,  "string_occupancy": True}),
]


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
    """
    Computes metrics over all notes (weighted average) and per string.
    Weighted average: each string's metric is weighted by its support (note count),
    so more frequent strings contribute proportionally more.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if string_labels is None:
        string_labels = sorted(set(y_true) | set(y_pred))

    # Overall: weighted average = each string weighted by its note count
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


# ── Pretty printing ──────────────────────────────────────────────────────────
def _format_filter_config(config):
    """Return a human-readable string showing which filter stages are active."""
    active  = [k for k, v in config.items() if v]
    return ", ".join(active) if active else "none"


def print_results_table(results, label="", filter_config=None):
    """Print a clean per-string and overall table to the console."""
    header = f"  Results: {label}" if label else "  Results"
    print(f"\n{'─' * 60}")
    print(header)
    if filter_config is not None:
        print(f"  Filter: {_format_filter_config(filter_config)}")
    print(f"{'─' * 60}")
    print(f"  {'String':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for lbl in results['string_labels']:
        m = results['per_string'][lbl]
        print(f"  {m['name']:<8} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    o = results['overall']
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Overall':<8} {o['precision']:>10.4f} {o['recall']:>10.4f} {o['f1']:>10.4f}")
    print(f"  {'Accuracy':<8} {o['accuracy']:>10.4f}")
    print(f"{'─' * 60}")


def print_total_table(results_solo, results_comp, filter_config=None):
    """Print the arithmetic mean of solo and comp metrics."""
    print(f"\n{'═' * 60}")
    print(f"  Total (arithmetic mean of Solo & Comp)")
    if filter_config is not None:
        print(f"  Filter: {_format_filter_config(filter_config)}")
    print(f"{'═' * 60}")
    print(f"  {'String':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for lbl in results_solo['string_labels']:
        ms = results_solo['per_string'][lbl]
        mc = results_comp['per_string'][lbl]
        p = (ms['precision'] + mc['precision']) / 2
        r = (ms['recall']    + mc['recall'])    / 2
        f = (ms['f1']        + mc['f1'])        / 2
        print(f"  {ms['name']:<8} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    os_ = results_solo['overall']
    oc  = results_comp['overall']
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Overall':<8} {(os_['precision']+oc['precision'])/2:>10.4f} "
          f"{(os_['recall']+oc['recall'])/2:>10.4f} "
          f"{(os_['f1']+oc['f1'])/2:>10.4f}")
    print(f"  {'Accuracy':<8} {(os_['accuracy']+oc['accuracy'])/2:>10.4f}")
    print(f"{'═' * 60}")


# ── Confusion matrix plots ────────────────────────────────────────────────────
def plot_combined_confusion_matrices(results_solo, results_comp, normalize=True, output_path=None):
    """
    Solo (top) and Comp (bottom) confusion matrices stacked vertically.
    Formatted for LaTeX: 3.52 in wide, Computer Modern serif, 7 pt.
    """
    rc = {
        'text.usetex':        True,
        'font.family':        'serif',
        'axes.labelsize':     10,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'axes.titlesize':     10,
    }

    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.40, 3.40))

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

            labels = results['string_names']

            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=vmax, ax=ax,
                annot_kws={'size': 9}, cbar=False,
                linewidths=0.5, linecolor='lightgray',
            )

            ax.set_xticklabels(labels, rotation=0, ha='center')
            ax.set_yticklabels(labels, rotation=0, va='center')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(title)

        plt.tight_layout(pad=0.5)

        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"\n[Plot] Saved combined confusion matrix → {output_path}")
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
                    stat_names=("median","mean","min","max","std","var","skewness","kurtosis","mode")):
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


def print_importance_table(imp_dict, title=""):
    """Print permutation importance as a clean table."""
    print(f"\n  {title}")
    print(f"  {'─'*50}")
    print(f"  {'Feature':<30} {'Δ Accuracy':>12} {'± Std':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*10}")
    for name, res in sorted(imp_dict.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {name:<30} {res['mean']:>12.4f} {res['std']:>10.4f}")
    print(f"  {'─'*50}")


def run_permutation(SVM, all_features, all_labels, all_valid_notes, audio_types):
    """Run grouped permutation importance and return results dicts."""
    FX_test     = np.stack(all_features)
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

    return seg_imp, stat_imp


def average_importance(imp_a, imp_b):
    """Arithmetic mean of two importance dicts."""
    merged = {}
    all_keys = set(imp_a.keys()) | set(imp_b.keys())
    for k in all_keys:
        a = imp_a.get(k, {"mean": 0.0, "std": 0.0})
        b = imp_b.get(k, {"mean": 0.0, "std": 0.0})
        merged[k] = {
            "mean": (a["mean"] + b["mean"]) / 2,
            "std":  (a["std"]  + b["std"])  / 2,
        }
    return merged


# ── Core evaluation run for one subset ───────────────────────────────────────
def run_subset(subset, SVM, filter_config, recalculate_features=False):
    config_test = ConfigParser()
    config_test.read(SUBSET_CONFIGS[subset])

    if recalculate_features:
        print(f"  Calculating features for {subset} (this may take a while) ...")
        calculate_features.main(config_test)

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
        print(f"  [{i}/{len(filepaths)}] {os.path.basename(filepath)}")

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

    return results, all_features, all_labels, all_valid_notes, audio_types


# ── Experiment 1: Classification ──────────────────────────────────────────────
def experiment_classification(SVM, recalc=False):
    """
    Classification with plausibility filter ON, on both solo and comp.
    Prints per-string and overall metrics, generates combined PDF plot.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT: Classification (with plausibility filter)")
    print("=" * 60)

    all_results = {}
    for subset in ("solo", "comp"):
        print(f"\n  Processing {subset} ...")
        results, *_ = run_subset(subset, SVM, FILTER_CONFIG_DEFAULT, recalculate_features=recalc)
        recalc = False  # only calculate once per subset, already done
        all_results[subset] = results
        print_results_table(results, label=subset.capitalize(), filter_config=FILTER_CONFIG_DEFAULT)

    print_total_table(all_results["solo"], all_results["comp"], filter_config=FILTER_CONFIG_DEFAULT)

    print("\n  Generating combined confusion matrix PDF ...")
    plot_combined_confusion_matrices(
        all_results["solo"],
        all_results["comp"],
        output_path="confusion_matrix_combined.pdf",
    )
    print("\n  Done.")


# ── Experiment 2: Feature Importance ──────────────────────────────────────────
def experiment_feature_importance(SVM, recalc=False):
    """
    Plausibility filter OFF. Measures feature contributions via permutation
    importance on solo and comp, then reports the arithmetic mean.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT: Feature Importance (no plausibility filter)")
    print("=" * 60)

    all_seg_imp  = {}
    all_stat_imp = {}
    all_results  = {}

    for subset in ("solo", "comp"):
        print(f"\n  Processing {subset} ...")
        results, features, labels, valid_notes, audio_types = run_subset(
            subset, SVM, FILTER_CONFIG_OFF, recalculate_features=recalc
        )
        recalc = False
        all_results[subset] = results
        print_results_table(results, label=subset.capitalize(), filter_config=FILTER_CONFIG_OFF)

        print(f"\n  Running permutation importance for {subset} ...")
        seg_imp, stat_imp = run_permutation(SVM, features, labels, valid_notes, audio_types)
        all_seg_imp[subset]  = seg_imp
        all_stat_imp[subset] = stat_imp

        print_importance_table(seg_imp,  title=f"Segment Importance – {subset.capitalize()}")
        print_importance_table(stat_imp, title=f"Stat Importance – {subset.capitalize()}")

    # Total: arithmetic mean
    print_total_table(all_results["solo"], all_results["comp"], filter_config=FILTER_CONFIG_OFF)

    avg_seg  = average_importance(all_seg_imp["solo"],  all_seg_imp["comp"])
    avg_stat = average_importance(all_stat_imp["solo"], all_stat_imp["comp"])

    print_importance_table(avg_seg,  title="Segment Importance – Total (mean of Solo & Comp)")
    print_importance_table(avg_stat, title="Stat Importance – Total (mean of Solo & Comp)")

    print("\n  Done.")


# ── Experiment 3: Plausibility Filter Importance ──────────────────────────────
def experiment_filter_importance(SVM, recalc=False):
    """
    Evaluates each filter in isolation, then all three combined:
      baseline               – no filter
      only physical_range    – importance = Δ F1 vs. baseline
      only centroid_penalty  – importance = Δ F1 vs. baseline
      only string_occupancy  – importance = Δ F1 vs. baseline
      all filters            – overall importance = Δ F1 vs. baseline
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT: Plausibility Filter Importance")
    print("=" * 60)

    summary   = {subset: {} for subset in ("solo", "comp")}
    first_run = True

    for stage_name, stage_config in FILTER_STAGES:
        active = [k for k, v in stage_config.items() if v]
        print(f"\n{'─' * 60}")
        print(f"  Filter stage: {stage_name}")
        print(f"  Active:       {active if active else 'none'}")
        print(f"{'─' * 60}")

        stage_results = {}
        for subset in ("solo", "comp"):
            print(f"\n  Processing {subset} ...")
            do_recalc = recalc and first_run
            results, *_ = run_subset(subset, SVM, stage_config, recalculate_features=do_recalc)
            first_run = False
            stage_results[subset] = results
            print_results_table(
                results,
                label=f"{subset.capitalize()} [{stage_name}]",
                filter_config=stage_config,
            )
            summary[subset][stage_name] = results['overall']['f1']

        print_total_table(stage_results["solo"], stage_results["comp"], filter_config=stage_config)

    # ── Summary table with Importance (Δ F1 vs. baseline) ─────────────────
    baseline_name = FILTER_STAGES[0][0]   # "no filter"
    all_name      = FILTER_STAGES[-1][0]  # "all filters"
    individual_stages = FILTER_STAGES[1:-1]  # the three isolated filters

    print(f"\n{'═' * 70}")
    print(f"  Summary: F1 and Importance (ΔF1 vs. baseline) per filter")
    print(f"{'═' * 70}")
    print(f"  {'Stage':<27} {'Solo':>7} {'Comp':>7} {'Total':>7}  {'Importance':>10}")
    print(f"  {'─'*27} {'─'*7} {'─'*7} {'─'*7}  {'─'*10}")

    # Baseline row (no importance delta)
    f1s = summary["solo"][baseline_name]
    f1c = summary["comp"][baseline_name]
    f1t = (f1s + f1c) / 2
    print(f"  {baseline_name:<27} {f1s:>7.4f} {f1c:>7.4f} {f1t:>7.4f}  {'(baseline)':>10}")

    # Individual filter rows
    baseline_solo  = summary["solo"][baseline_name]
    baseline_comp  = summary["comp"][baseline_name]
    baseline_total = (baseline_solo + baseline_comp) / 2

    for stage_name, _ in individual_stages:
        f1s   = summary["solo"][stage_name]
        f1c   = summary["comp"][stage_name]
        f1t   = (f1s + f1c) / 2
        delta = f1t - baseline_total
        sign  = "+" if delta >= 0 else ""
        print(f"  {stage_name:<27} {f1s:>7.4f} {f1c:>7.4f} {f1t:>7.4f}  {sign}{delta:>9.4f}")

    # All-filters row (overall importance)
    f1s   = summary["solo"][all_name]
    f1c   = summary["comp"][all_name]
    f1t   = (f1s + f1c) / 2
    delta = f1t - baseline_total
    sign  = "+" if delta >= 0 else ""
    print(f"  {'─'*27} {'─'*7} {'─'*7} {'─'*7}  {'─'*10}")
    print(f"  {all_name:<27} {f1s:>7.4f} {f1c:>7.4f} {f1t:>7.4f}  {sign}{delta:>9.4f}  ← gesamt")
    print(f"{'═' * 70}")

    print("\n  Done.")

# ── Main ──────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "classification":    experiment_classification,
    "feature-importance": experiment_feature_importance,
    "filter-importance":  experiment_filter_importance,
}


def main():
    parser = argparse.ArgumentParser(
        description="SVM Evaluation – GuitarSet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
experiments:
  classification       Classification with plausibility filter + PDF plot
  feature-importance   Permutation feature importance (no filter)
  filter-importance    Evaluate incremental plausibility filter stages
        """,
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run.",
    )
    parser.add_argument(
        "--recalculate-features",
        action="store_true",
        default=None,
        help="Recalculate features before evaluation (expensive, only needed once).",
    )
    args = parser.parse_args()

    # Interactive selection: experiment
    if args.experiment is None:
        print("\nSelect experiment:")
        print("  [1] classification         – Classification with plausibility filter + PDF plot")
        print("  [2] feature-importance     – Permutation feature importance (no filter)")
        print("  [3] filter-importance      – Evaluate incremental plausibility filter stages")
        choice = input("\nEnter 1, 2, or 3: ").strip()
        args.experiment = {
            "1": "classification",
            "2": "feature-importance",
            "3": "filter-importance",
        }.get(choice)
        if args.experiment is None:
            print("Invalid choice.")
            sys.exit(1)

    # Interactive selection: recalculate features
    if args.recalculate_features is None:
        choice = input("\nRecalculate features? (expensive, only needed once) [y/N]: ").strip().lower()
        args.recalculate_features = choice == "y"

    SVM = joblib.load("svm_full.joblib")
    EXPERIMENTS[args.experiment](SVM, recalc=args.recalculate_features)
    print("\n\nSuccess!")

if __name__ == "__main__":
    main()