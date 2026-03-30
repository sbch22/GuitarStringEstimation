# import
import os
import sys
sys.path.append(os.path.abspath(''))

from configparser import ConfigParser
import joblib
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

# import functions
import calculate_features
import find_partials
from utils.Track_dataclass import filter_analysis


def get_fret(midi_pitch, string_index):
    open_string_midi = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    return midi_pitch - open_string_midi[string_index]  # 0 = Leersaite, 20 = 20. Bund


# Centroid: gewichteter Mittelwert der letzten N Bund-Positionen
class FretboardCentroid:
    def __init__(self, window_size=5, max_hand_span=6):
        self.window_size = window_size
        self.max_hand_span = max_hand_span  # Bünde, die eine Hand bequem greifen kann
        self.recent_frets = []  # [(fret_position, timestamp)]

    def update(self, fret, timestamp):
        self.recent_frets.append((fret, timestamp))
        if len(self.recent_frets) > self.window_size:
            self.recent_frets.pop(0)

    def get_centroid(self, current_time):
        if not self.recent_frets:
            return None

        # Zeitlicher Decay: ältere Noten haben weniger Gewicht
        weights = []
        for fret, timestamp in self.recent_frets:
            age = current_time - timestamp
            weight = np.exp(-age * 2.0)  # Decay-Rate anpassbar
            weights.append(weight)

        weights = np.array(weights)
        frets = np.array([f for f, _ in self.recent_frets])
        return np.average(frets, weights=weights)


# ── String Occupancy: welche Saiten sind im aktuellen Zeitfenster belegt? ──
# Eine Saite kann nicht zwei Noten gleichzeitig spielen
def get_occupied_strings(current_note, all_notes, all_probs, tolerance=0.05):
    occupied = set()
    current_onset  = current_note.attributes.onset
    current_offset = current_note.attributes.offset

    for other_note, other_prob in zip(all_notes, all_probs):
        if other_note is current_note:
            continue
        other_onset  = other_note.attributes.onset
        other_offset = other_note.attributes.offset

        # Zeitliche Überlappung?
        overlaps = (other_onset < current_offset - tolerance and
                    other_offset > current_onset + tolerance)
        if overlaps:
            occupied.add(int(np.argmax(other_prob)))  # aktuell wahrscheinlichste Saite
    return occupied


# Filter-Toggles
FILTER_CONFIG = {
    "physical_range":   True,   # Unmögliche Saiten auf 0
    "fret_probability": False, # worse
    "centroid_penalty": True,   # Hand-Position / Spannweite
    "string_occupancy": True,   # Keine zwei Noten gleichzeitig auf einer Saite
}


def plausibility_filter(probabilities, notes, centroid_tracker, config=FILTER_CONFIG):
    string_ranges = {
        0: (40, 60), 1: (45, 65), 2: (50, 70),
        3: (55, 75), 4: (59, 79), 5: (64, 84),
    }

    fret_frequency_GOAT = np.array([
        [3646, 1033, 1162, 1727, 799, 1294, 417, 825, 528, 657, 184, 207, 236, 66, 26, 70, 11, 8, 0, 8],
        [3888, 304, 2546, 2027, 1758, 2421, 1559, 2435, 905, 1521, 544, 937, 278, 117, 244, 62, 38, 20, 0, 0],
        [4488, 263, 2922, 1070, 1562, 1915, 1964, 2603, 1220, 2191, 950, 1631, 611, 668, 523, 97, 53, 67, 11, 28],
        [3452, 499, 2507, 759, 2309, 1698, 1542, 1939, 1630, 2414, 1484, 915, 1326, 369, 654, 153, 156, 141, 75, 79],
        [3100, 456, 1031, 1728, 882, 1530, 786, 1267, 754, 1155, 982, 956, 779, 709, 472, 508, 109, 305, 83, 73],
        [1789, 208, 1078, 757, 664, 416, 250, 578, 202, 386, 417, 432, 571, 261, 261, 174, 145, 220, 20, 65],
    ])

    fret_prob_GOAT = fret_frequency_GOAT / fret_frequency_GOAT.sum(axis=1, keepdims=True)

    filtered_probabilities = []

    for note, prob in zip(notes, probabilities):
        midi_pitch = note.attributes.midi_note
        timestamp  = note.attributes.onset
        masked_prob = prob.copy()

        # ── 1. Physical Range Filter ──────────────────────────────────────────
        if config["physical_range"]:
            for string in range(len(prob)):
                low, high = string_ranges[string]
                if not (low <= midi_pitch <= high):
                    masked_prob[string] = 0.0

        # ── 1.5 Probabilities per Fret Filter (GOAT) ──────────────────────────────────────────
        if config["fret_probability"]:
            for string in range(len(prob)):
                if masked_prob[string] == 0.0:
                    continue  # bereits physikalisch ausgeschlossen

                fret = get_fret(midi_pitch, string)  # welcher Bund wäre das auf dieser Saite?

                if 0 <= fret < fret_prob_GOAT.shape[1]:  # innerhalb der 20 Bünde
                    masked_prob[string] *= fret_prob_GOAT[string, fret]
                else:
                    masked_prob[string] = 0.0  # außerhalb → unmöglich

        # ── 2. Centroid Penalty ───────────────────────────────────────────────
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

        # ── 3. String Occupancy ───────────────────────────────────────────────
        if config["string_occupancy"]:
            occupied = get_occupied_strings(note, notes, filtered_probabilities or probabilities)
            for string in occupied:
                masked_prob[string] *= 0.05  # weich bestrafen statt hart auf 0

        # Renormalisieren nach jeder Note
        total = masked_prob.sum()
        if total > 0:
            masked_prob /= total
        else:
            masked_prob = prob.copy()  # Fallback: Original-Probs

        # Centroid updaten
        best_string = np.argmax(masked_prob)
        best_fret   = get_fret(midi_pitch, best_string)
        centroid_tracker.update(best_fret, timestamp)

        filtered_probabilities.append(masked_prob)

    probabilities = np.array(filtered_probabilities)

    return probabilities

def plot_confusion_matrix(results, normalize=True, title='Gitarrensaitenklassifikation', subset_label=None):
    cm     = results['confusion_matrix'].copy().astype(float)
    labels = results['string_names']
    full_title = f"{title} ({subset_label})" if subset_label else title


    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = '.2f', 1.0
    else:
        fmt, vmax = 'd', None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=vmax, ax=ax
    )

    if normalize:
        for text in ax.texts:
            text.set_text(text.get_text())
    ax.set_xlabel('Saite (geschätzt)')
    ax.set_ylabel('Saite (Grundwahrheit)')
    ax.set_title(full_title)
    plt.tight_layout()
    plt.show()


STRING_NAMES = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # index → name mapping

def evaluate_classification(y_true, y_pred, string_labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Use integer indices that actually exist in the data
    if string_labels is None:
        string_labels = sorted(set(y_true) | set(y_pred))  # e.g. [0,1,2,3,4,5]

    # Overall metrics
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')

    # Per-string metrics (average=None returns one value per label)
    per_string_precision = precision_score(y_true, y_pred, average=None, labels=string_labels)
    per_string_recall    = recall_score(y_true, y_pred, average=None, labels=string_labels)
    per_string_f1        = f1_score(y_true, y_pred, average=None, labels=string_labels)

    per_string_metrics = {}
    for i, label in enumerate(string_labels):
        mask = y_true == label
        per_string_metrics[label] = {
            'name':      STRING_NAMES[label],          # human-readable name
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
        'string_labels':    string_labels,   # [0,1,2,3,4,5]
        'string_names':     [STRING_NAMES[l] for l in string_labels],  # ['E2',...]
    }

def get_stat_groups(groups, measure_segments_with_k, stat_names=["median","mean","min","max","std","var", "skewness", "kurtosis", "mode"]):
    """
    For measure arrays (shape 9 x K), extract indices for each stat across all measure segments.
    measure_segments_with_k: dict of {seg_name: n_partials}, e.g.:
        {
            "rel_partial_amplitudes": 25,
            "rel_freq_deviations":    24,
        }
    """
    stat_groups = {name: [] for name in stat_names}

    for seg_name, k in measure_segments_with_k.items():
        seg_indices = groups[seg_name]
        stat_index_blocks = np.array(seg_indices).reshape(len(stat_names), k)
        for i, stat_name in enumerate(stat_names):
            stat_groups[stat_name].extend(stat_index_blocks[i].tolist())

    return stat_groups


def get_feature_groups(sample_fv):
    """
    Build index groups by reconstructing segment boundaries from a sample Features object.

    Segment sizes based on actual dimensions:
        f0:                     1           (scalar)
        betas_measures:         9           (1D: one stat per beta coefficient)
        valid_partials:         25          (1D: valid flag per partial)
        spectral_centroid:      9           (1D: one stat per frame)
        rel_partial_amplitudes: 9 x 25 = 225  (9 stats × 25 partials)
        amp_decay_coefficients: 25          (1D: one decay coeff per partial)
        rel_freq_deviations:    9 x 24 = 216  (9 stats × 24 partials)

        510 Features overall
    """
    segments = {
        "f0":                     1,
        "betas_measures":         sample_fv.betas_measures.size,        # 9
        "valid_partials":         sample_fv.valid_partials.size,         # 25
        "spectral_centroid":      sample_fv.spectral_centroid.size,      # 9
        "rel_partial_amplitudes": sample_fv.rel_partial_amplitudes.size, # 225
        "amp_decay_coefficients": sample_fv.amp_decay_coefficients.size, # 25
        "rel_freq_deviations":    sample_fv.rel_freq_deviations.size,    # 216
    }

    groups = {}
    offset = 0
    for name, size in segments.items():
        groups[name] = list(range(offset, offset + size))
        offset += size

    return groups


def get_partial_groups(groups, sample_fv):
    """
    Group feature indices by partial index k, across all segments with a partial axis.

    Segments included:
        rel_partial_amplitudes  (9, 25): stat × partial
        amp_decay_coefficients  (25,):   flat, one per partial
        rel_freq_deviations     (9, 24): stat × partial  — only 24 partials!

    Returns dict: {k: [feature_idx, ...]} for k in 0..max_partial
    """
    n_stats = 9

    amp_indices = np.array(groups["rel_partial_amplitudes"]).reshape(n_stats, sample_fv.rel_partial_amplitudes.shape[
        1])  # (9, 25)
    decay_indices = np.array(groups["amp_decay_coefficients"])  # (25,)
    freq_indices = np.array(groups["rel_freq_deviations"]).reshape(n_stats,
                                                                   sample_fv.rel_freq_deviations.shape[1])  # (9, 24)

    n_partials_max = decay_indices.shape[0]  # 25

    partial_groups = {}
    for k in range(n_partials_max):
        idx = []
        idx.extend(amp_indices[:, k].tolist())  # 9 indices from rel_partial_amplitudes
        idx.append(decay_indices[k].item())  # 1 index  from amp_decay_coefficients
        if k < freq_indices.shape[1]:  # rel_freq_deviations only has 24
            idx.extend(freq_indices[:, k].tolist())  # 9 indices from rel_freq_deviations
        partial_groups[k] = idx

    return partial_groups


def grouped_permutation_importance(model, X, y, groups, n_repeats=5, scoring="accuracy", random_state=42):
    """
    groups: dict of {group_name: [col_idx, col_idx, ...]}
    Returns dict of {group_name: {"mean": float, "std": float}}
    """
    rng = np.random.RandomState(random_state)
    # Baseline score
    baseline = accuracy_score(y, model.predict(X))

    results = {}
    for group_name, col_indices in groups.items():
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Permute all columns in this group with the SAME shuffle order
            shuffle_idx = rng.permutation(len(y))
            X_permuted[:, col_indices] = X_permuted[shuffle_idx][:, col_indices]
            score = accuracy_score(y, model.predict(X_permuted))
            scores.append(baseline - score)  # importance = drop in accuracy

        results[group_name] = {
            "mean": np.mean(scores),
            "std":  np.std(scores),
        }

    return results



def main(subset):
    # load configs
    if subset == 'comp':
        config_test = ConfigParser()
        config_test.read('configs/config_test_comp.ini')
    elif subset == 'solo':
        config_test = ConfigParser()
        config_test.read('configs/config_test_solo.ini')
    elif subset == 'GOAT':
        config_test = ConfigParser()
        config_test.read('configs/config_test_GOAT.ini')
    elif subset == 'single_note_IDMT':
        config_test = ConfigParser()
        config_test.read('configs/config_test_single_note_IDMT.ini')
    elif subset == 'IDMT':
        config_test = ConfigParser()
        config_test.read('configs/config_test_IDMT.ini')


    # calculate_features.main(config_test)


    track_directory = config_test.get('paths', 'track_directory')
    audio_types_raw = config_test.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]

    # SVM = joblib.load("SVM_full_2203_solo.joblib")
    SVM = joblib.load("SVM_full-1_new.joblib")


    # TODO: suppress warnings empty features
    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values",
        category=UserWarning,
        module="sklearn.impute"
    )

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory,filename)
        for filename in os.listdir(track_directory)
        if filename.endswith(".pkl")
        if os.path.isfile(os.path.join(track_directory, filename))
        # if "mono" in filename
    ]

    all_predictions = []
    all_features = []
    all_labels = []
    all_notes = []
    all_valid_notes = []

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Classification:  {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)

        # reset centroid tracker per track
        centroid_tracker = FretboardCentroid()

        for note in track.notes:
            all_notes.append(note)

        for audio_type in audio_types:
            track_fvs_for_type = []
            track_notes_for_type = []
            for note in track.valid_notes:
                if audio_type not in note.features:
                    continue
                fv = note.features[audio_type].feature_vector
                if fv is None:
                    continue
                track_fvs_for_type.append(fv)
                track_notes_for_type.append(note)
                all_features.append(fv)
                all_labels.append(note.attributes.string_index)
                all_valid_notes.append(note)

            if not track_fvs_for_type:
                continue

            track_predictions = SVM.predict_proba(np.stack(track_fvs_for_type))
            track_predictions = plausibility_filter(track_predictions, track_notes_for_type, centroid_tracker)
            all_predictions.extend(track_predictions)


    """ Filter analysis testSet"""
    filter_analysis(all_notes)

    string_labels = [0, 1, 2, 3, 4, 5]

    # """ Permutation """
    # FX_test = np.stack(all_features)
    # labels_test = np.array(all_labels)
    #
    # sample_fv = all_valid_notes[0].features[audio_types[0]]
    # feature_groups = get_feature_groups(sample_fv)
    #
    # # Per-segment K values for the (6, K) measure arrays only
    # measure_segs_with_k = {
    #     "rel_partial_amplitudes": sample_fv.rel_partial_amplitudes.shape[1],  # 25
    #     "rel_freq_deviations": sample_fv.rel_freq_deviations.shape[1],  # 24
    # }
    # stat_groups = get_stat_groups(feature_groups, measure_segs_with_k)
    #
    # # Run both
    # segment_importance = grouped_permutation_importance(SVM, FX_test, labels_test, feature_groups, n_repeats=5)
    # stat_importance = grouped_permutation_importance(SVM, FX_test, labels_test, stat_groups, n_repeats=5)
    #
    # for name, res in sorted(segment_importance.items(), key=lambda x: -x[1]["mean"]):
    #     print(f"{name:30s}  importance: {res['mean']:.4f} ± {res['std']:.4f}")
    #
    # for name, res in sorted(stat_importance.items(), key=lambda x: -x[1]["mean"]):
    #     print(f"{name:30s}  importance: {res['mean']:.4f} ± {res['std']:.4f}")

    """ Overall (tracks) """
    results = evaluate_classification(all_labels, np.argmax(all_predictions, axis=1), string_labels=string_labels)

    print(results['overall'])  # Overall accuracy, F1, etc.
    print(results['per_string'])  # Per-string breakdown
    # plot_confusion_matrix(results, subset)  # Heatmap of string confusions

    # Aufruf:
    plot_confusion_matrix(results, subset_label=subset)


    print("\n\n Success! Classification done.")

if __name__ == "__main__":
    # test on the following subsets:
    subset_solo = 'solo'
    subset_comp = 'comp'
    subset_GOAT = 'GOAT'
    subset_single_note_IDMT = 'single_note_IDMT'
    subset_IDMT = 'IDMT'

    main(subset_solo)
    main(subset_comp)
    # main(subset_GOAT)
    # main(subset_single_note_IDMT)
    # main(subset_IDMT)
