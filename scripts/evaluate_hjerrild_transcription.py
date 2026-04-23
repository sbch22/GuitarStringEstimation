import os
import sys
sys.path.append(os.path.abspath(''))

import pickle
import csv
from configparser import ConfigParser
import numpy as np

ONSET_TOLERANCE_SEC = 0.05  # ±50ms

# Standard tuning open string MIDI notes (string index 1–6, low E to high E)
STANDARD_TUNING_MIDI = {
    0: 40,  # E2
    1: 45,  # A2
    2: 50,  # D3
    3: 55,  # G3
    4: 59,  # B3
    5: 64,  # E4
}

def fret_to_midi(string_index: int, fret: int) -> int:
    """Calculate MIDI note number from guitar string (1=low E) and fret in standard tuning."""
    return STANDARD_TUNING_MIDI[string_index] + fret


def load_transcription_csv(filepath):
    """Load prediction CSV into list of dicts with onset, string, fret, and midi_note."""
    notes = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            string_0 = int(row['string_estimate']) - 1  # convert 1–6 → 0–5 once
            fret     = int(row['fret_estimate'])
            notes.append({
                'onset':     float(row['onset_sec']),
                'string':    string_0,
                'fret':      fret,
                'midi_note': fret_to_midi(string_0, fret),  # 0-based, matches STANDARD_TUNING_MIDI
            })
    return notes


def evaluate_onset_pitch(gt_notes, pred_notes, onset_tol=ONSET_TOLERANCE_SEC):
    """
    Onset + pitch (MIDI) evaluation on all gt_notes.
    Match requires: onset within tolerance AND midi_note matches.
    Returns (tp, fp, fn).
    """
    matched_pred_indices = set()
    tp = 0

    for gt_note in gt_notes:
        gt_onset     = gt_note.attributes.onset
        gt_midi_note = gt_note.attributes.midi_note

        candidates = [
            (i, pred) for i, pred in enumerate(pred_notes)
            if i not in matched_pred_indices
            and abs(pred['onset'] - gt_onset) <= onset_tol
            and pred['midi_note'] == gt_midi_note
        ]

        if candidates:
            best_idx, _ = min(candidates, key=lambda x: abs(x[1]['onset'] - gt_onset))
            matched_pred_indices.add(best_idx)
            tp += 1

    fp = len(pred_notes) - len(matched_pred_indices)
    fn = len(gt_notes)   - tp
    return tp, fp, fn


def evaluate_string(valid_notes, pred_notes, onset_tol=ONSET_TOLERANCE_SEC):
    """
    String classification evaluation on valid_notes (GT string is known).

    Step 1 — match by onset + midi_note (same as onset/pitch eval).
    Step 2 — among matched pairs, check if string also matches.

      - onset+pitch match AND correct string  → TP
      - onset+pitch match BUT wrong string    → FN (missed string) + FP (wrong string pred)
      - no onset+pitch match at all           → FN
      - pred not matched to any valid_note    → FP

    Returns (tp, fp, fn).
    """
    matched_pred_indices = set()
    tp = 0
    wrong_string = 0

    for gt_note in valid_notes:
        gt_onset     = gt_note.attributes.onset
        gt_midi_note = gt_note.attributes.midi_note
        gt_string    = gt_note.attributes.string_index

        candidates = [
            (i, pred) for i, pred in enumerate(pred_notes)
            if i not in matched_pred_indices
            and abs(pred['onset'] - gt_onset) <= onset_tol
            and pred['midi_note'] == gt_midi_note
        ]

        if candidates:
            best_idx, best_pred = min(candidates, key=lambda x: abs(x[1]['onset'] - gt_onset))
            matched_pred_indices.add(best_idx)
            if best_pred['string'] == gt_string:
                tp += 1
            else:
                wrong_string += 1  # pitch matched but string wrong

    # FN = unmatched valid notes + pitch-matched but wrong-string notes
    fn = (len(valid_notes) - len(matched_pred_indices)) + wrong_string
    # FP = unmatched predictions + wrong-string predictions
    fp = (len(pred_notes) - len(matched_pred_indices)) + wrong_string
    return tp, fp, fn


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def main(subset):
    config_test = ConfigParser()
    if subset == 'comp':
        config_test.read('../gse/src/configs/config_test_comp.ini')
    elif subset == 'solo':
        config_test.read('../gse/src/configs/config_test_solo.ini')

    track_directory         = config_test.get('paths', 'track_directory')
    track_directory = track_directory.replace('../', '', 1)


    """ Choose between standard output (/martin) and same model with YMT3 Onsets (/martin/YMT3_Onsets)"""
    # transcription_directory = '../data/GuitarSet/hjerrild_transcription/audio_mono-mic/martin/'
    transcription_directory = '../data/GuitarSet/hjerrild_et_al_eval/hjerrild_transcription/audio_mono-mic/martin/YMT3_onsets_and_f0'    # transcription_directory = '../data/GuitarSet/hjerrild_transcription/audio_mono-mic/GuitarSet/'



    filepaths = sorted([
        os.path.join(track_directory, fn)
        for fn in os.listdir(track_directory)
        if fn.endswith(".pkl") and os.path.isfile(os.path.join(track_directory, fn))
    ])

    # Accumulators for both evaluations
    pitch_tp = pitch_fp = pitch_fn = 0
    str_tp   = str_fp   = str_fn   = 0
    skipped  = 0

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            track = pickle.load(f)

        filename = os.path.basename(filepath)
        filename = os.path.splitext(filename)[0]
        filename = filename.replace('_track', '_mic_transcription.csv')
        transcription_filepath = os.path.join(transcription_directory, filename)

        if not os.path.isfile(transcription_filepath):
            print(f"  WARNING: transcription not found – {transcription_filepath}, skipping.")
            skipped += 1
            continue

        pred_notes  = load_transcription_csv(transcription_filepath)
        gt_notes    = [n for n in track.notes if n.origin == 'gt']
        valid_notes = track.valid_notes

        # --- Onset + Pitch F1 (all GT notes) ---
        tp, fp, fn  = evaluate_onset_pitch(gt_notes, pred_notes)
        pitch_tp   += tp
        pitch_fp   += fp
        pitch_fn   += fn

        # --- String F1 (valid notes only) ---
        tp, fp, fn = evaluate_string(valid_notes, pred_notes)
        str_tp    += tp
        str_fp    += fp
        str_fn    += fn

    # Print combined summary
    p_p, p_r, p_f1 = compute_metrics(pitch_tp, pitch_fp, pitch_fn)
    s_p, s_r, s_f1 = compute_metrics(str_tp,   str_fp,   str_fn)

    print(f"\n{'='*60}")
    print(f"Subset: {subset.upper()}  ({len(filepaths) - skipped}/{len(filepaths)} tracks evaluated)")
    print(f"\n  Onset + Pitch  (all GT notes: {pitch_tp + pitch_fn})")
    print(f"    TP: {pitch_tp:4d}  FP: {pitch_fp:4d}  FN: {pitch_fn:4d}")
    print(f"    Precision: {p_p:.4f}  Recall: {p_r:.4f}  F1: {p_f1:.4f}")
    print(f"\n  String         (valid_notes only: {str_tp + str_fn})")
    print(f"    TP: {str_tp:4d}  FP: {str_fp:4d}  FN: {str_fn:4d}")
    print(f"    Precision: {s_p:.4f}  Recall: {s_r:.4f}  F1: {s_f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main('solo')
    main('comp')