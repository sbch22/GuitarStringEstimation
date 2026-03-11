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
    "fret_probability": True, # fret probabilities from GOAT
    "centroid_penalty": True,   # Hand-Position / Spannweite
    "string_occupancy": False,   # Keine zwei Noten gleichzeitig auf einer Saite
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

        # Renormalisieren nach jedem Note
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


def plot_confusion_matrix(results, normalize=True, title='Guitar String Confusion Matrix'):
    cm     = results['confusion_matrix']
    labels = results['string_names']   # ← use names here instead of indices

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
    ax.set_xlabel('Predicted String')
    ax.set_ylabel('True String')
    ax.set_title(title)
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


def main():
    # load configs
    config_test = ConfigParser()
    config_test.read('config_test.ini')

    config_train = ConfigParser()
    config_train.read('config_train.ini')

    find_partials.main(
        config_test
    )

    calculate_features.main(
        config_test
    )

    centroid_tracker = FretboardCentroid()

    track_directory = config_test.get('paths', 'track_directory')

    SVM = joblib.load("svm_pipeline.joblib")

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    all_predictions = []
    all_features = []
    all_labels = []
    all_notes = []

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Processing {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)


        notes_by_track = defaultdict(list)
        track_feature_vectors = []
        track_labels = []

        for note in track.notes:
            all_notes.append(note)

        for note in track.valid_notes:
            track_feature_vectors.append(note.features.feature_vector)
            track_labels.append(note.attributes.string_index)  # GT
            notes_by_track[os.path.basename(filepath)].append(note)

            fv = note.features.feature_vector
            all_features.append(fv)
            all_labels.append(note.attributes.string_index)

        # stack over all tracks
        FX_track_test = np.stack(track_feature_vectors)  # (total_notes, N)
        labels_track_test_gt = np.array(track_labels) # (total_notes,)

        probs = SVM.predict_proba(FX_track_test)
        # assign probs to notes
        for i, note in enumerate(track.valid_notes):
            note.string_probs = probs[i,:]

        track_predictions = plausibility_filter(probs, track.valid_notes, centroid_tracker)
        all_predictions.append(track_predictions)

    filter_analysis(all_notes)

    X_test = np.stack(all_features)
    y_test = np.array(all_labels)

    string_labels = [0, 1, 2, 3, 4, 5]

    """ Permutation """
    result = permutation_importance(
        SVM,
        X_test,
        y_test,
        n_repeats=10,
        scoring="accuracy",
        n_jobs=-1,
    )

    importance = result.importances_mean
    ranking = np.argsort(importance)[::-1]
    for i in ranking:
        print(f"Feature {i}: {importance[i]:.5f}")


    predictions = np.concatenate(all_predictions, axis=0)  # shape: (total_notes, n_classes)
    results = evaluate_classification(all_labels, np.argmax(predictions, axis=1), string_labels=string_labels)

    print(results['overall'])  # Overall accuracy, F1, etc.
    print(results['per_string'])  # Per-string breakdown
    plot_confusion_matrix(results)  # Heatmap of string confusions

    print("\n\n Success! Classification done.")

if __name__ == "__main__":
    main()