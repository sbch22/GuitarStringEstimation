import os
import sys
from typing import List, Tuple, Dict, Optional
import argparse
import math
import json
import dataclasses

import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import torch

from amt.src.betaDistributions import filter_outliers_iqr
# Project-specific imports (unchanged)
from utils.note_event_dataclasses import matchNote, stringNote

from betaDistributions import (
    noteToFreq,
    flatten_recursive,
    calculate_partials,
    calculate_fundamental,
    estimate_inharmonicity_coefficients,
    transcribe_notes,
    prepare_media,
    load_model_checkpoint,
    filter_outliers_iqr
)
from string_classification import (
    wasserstein,
    wasserstein_freq_semi_empirical,
    freq_theoretical,
    freq_semi_empirical
)

# Utilities
def get_annotation_filename(audio_filename: str) -> str:
    """
    Derive the annotation filename for a given audio filename.

    The function removes audio-specific suffixes and any "_pshift" marker,
    then appends "_matchedNotes.npy".

    Args:
        audio_filename: The audio filename (e.g., 'piece_solo_cln.wav').

    Returns:
        The corresponding annotation filename (e.g., 'piece_solo_matchedNotes.npy').
    """
    base_name = os.path.splitext(audio_filename)[0]

    # Remove pitch-shift marker and common suffixes
    if "_pshift" in base_name:
        base_name = base_name.split("_pshift")[0]

    for suffix in ["_hex", "_cln", "_mic", "_debleeded"]:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]

    return f"{base_name}_matchedNotes.npy"


def load_betas_from_json(filename: str) -> Dict:
    """
    Load Beta curve data (dictionary) from a JSON file.

    Args:
        filename: Path to JSON file.

    Returns:
        Deserialized JSON dictionary.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def compute_kde_normalized(values: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """
    Compute a kernel density estimate for `values` on the grid `x_vals`, normalized to unit area.

    Args:
        values: 1D array of samples.
        x_vals: Grid to evaluate the KDE.

    Returns:
        KDE evaluated on x_vals normalized such that area under the curve == 1.
    """
    if len(values) == 0:
        return np.zeros_like(x_vals)

    kde = gaussian_kde(values)
    kde_vals = kde(x_vals)
    area = np.trapz(kde_vals, x_vals)
    if area <= 0:
        return np.zeros_like(kde_vals)
    return kde_vals / area


def extract_GT(annotation_filename: str, annotation_directory: str = '../../data/guitarset_yourmt3_16k/annotation/') -> np.ndarray:
    """
    Load ground-truth matched notes from a .npy annotation file.

    Args:
        annotation_filename: Filename (with extension) of annotation.
        annotation_directory: Directory to look for annotations.

    Returns:
        Numpy array with annotation note objects.

    Raises:
        AssertionError if the file does not exist.
    """
    annotation_filepath = os.path.join(annotation_directory, annotation_filename)
    if not os.path.exists(annotation_filepath):
        raise FileNotFoundError(f"Annotation file not found: {annotation_filepath}")
    notes = np.load(annotation_filepath, allow_pickle=True)
    return notes


# Plotting / Analysis
def plot_frequency_distribution(string_notes: List[stringNote]) -> None:
    """
    Plot per-string frequency histogram based on the pitch of notes that have stringGT defined.

    Args:
        string_notes: List of stringNote objects
    """
    string_frequencies = {
        1: 82.41,
        2: 110.00,
        3: 146.83,
        4: 196.00,
        5: 246.94,
        6: 329.63,
    }

    frequencies_by_string = {i: [] for i in range(6)}
    for note in string_notes:
        if note.stringGT is not None:
            freq = noteToFreq(note.pitch)
            # If stringGT is in the 1..6 range store under 0..5 index (original code treated strings 0..5)
            try:
                frequencies_by_string[int(note.stringGT)].append(freq)
            except Exception:
                # ignore invalid mapping
                continue

    plt.figure(figsize=(10, 6))
    for s_idx, freqs in frequencies_by_string.items():
        if freqs:
            plt.hist(freqs, bins=20, alpha=0.5, label=f'String {s_idx}', density=True)
    plt.title("Frequency distribution per string")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_string_predictions(string_notes: List[stringNote]) -> Dict:
    """
    Compute accuracy, confusion matrix, per-string accuracy and mean likelihood ratio.

    Args:
        string_notes: list of stringNote objects with fields:
            - string_pred (predicted),
            - stringGT (ground-truth),
            - likelihood_ratio (optional)

    Returns:
        Dictionary containing accuracy, confusion_matrix (np.ndarray 6x6), string_counts,
        mean_likelihood, per_string_accuracy (dict).
    """
    total_notes = 0
    correct_predictions = 0
    likelihoods = []
    conf_matrix = np.zeros((6, 6), dtype=int)
    preds = []
    gts = []

    for note in string_notes:
        if note.string_pred is None or note.stringGT is None:
            continue
        gt = int(note.stringGT)
        pred = int(note.string_pred)
        # guard against out-of-range indices
        if not (0 <= gt < 6 and 0 <= pred < 6):
            continue
        conf_matrix[gt, pred] += 1
        preds.append(pred)
        gts.append(gt)
        total_notes += 1
        if gt == pred:
            correct_predictions += 1
        if note.likelihood_ratio is not None:
            likelihoods.append(note.likelihood_ratio)

    if total_notes == 0:
        print("No notes with both string_pred and stringGT were found.")
        return {
            "accuracy": 0.0,
            "confusion_matrix": None,
            "string_counts": None,
            "mean_likelihood": 0.0,
            "per_string_accuracy": None
        }

    accuracy = (correct_predictions / total_notes) * 100.0
    print(f"Overall Accuracy (String Prediction): {accuracy:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(6), yticklabels=np.arange(6))
    plt.xlabel("Predicted String")
    plt.ylabel("True String")
    plt.title("Confusion Matrix (String predictions)")
    plt.show()

    unique, counts = np.unique(preds, return_counts=True)
    string_counts = dict(zip(unique.tolist(), counts.tolist()))
    mean_likelihood = float(np.mean(likelihoods)) if likelihoods else 0.0

    correct_counts = {i: int(conf_matrix[i, i]) for i in range(6)}
    total_counts = {i: int(np.sum(conf_matrix[i, :])) for i in range(6)}
    per_string_accuracy = {
        i: (correct_counts[i] / total_counts[i]) * 100.0 if total_counts[i] > 0 else 0.0 for i in range(6)
    }

    print("Per-string accuracy (%)")
    for s, acc in per_string_accuracy.items():
        print(f"String {s}: {acc:.2f}%")

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "string_counts": string_counts,
        "mean_likelihood": mean_likelihood,
        "per_string_accuracy": per_string_accuracy
    }


# Beta Estimation
def estimate_betas_for_note(string_sig: np.ndarray, sr: int, notes: List[stringNote], dbg: bool = False) -> List[stringNote]:
    """
    Legacy Beta estimation kept as-is (with small readability improvements).

    For each note, slice the signal between onset/offset, frame it using sliding windows,
    compute FFT, detect peaks, estimate the fundamental via phase, extract partials,
    compute beta per partial and apply an IQR filter.

    Args:
        string_sig: Full audio signal (1D numpy array)
        sr: Sampling rate (Hz)
        notes: List of stringNote instances (with onset, offset, pitch)
        dbg: Debug flag; when True, plot spectra for each frame

    Returns:
        The input notes list updated in-place with .noteBetas set to a flattened list of beta values.
    """
    beta_max = 2e-4

    for idx, note in enumerate(notes):
        try:
            onset_sample = int(note.onset * sr + 0.05 * sr)
            offset_sample = int(note.offset * sr)
            note_freq = noteToFreq(note.pitch)
            note_sig = string_sig[onset_sample:offset_sample]
        except Exception:
            # keep original behavior (skip bad indices)
            continue

        # STFT-like framing parameters (kept as in original)
        W = 2048
        H = W // 8
        fft_size = W

        if len(note_sig) <= W:
            # Not enough samples to compute frames
            continue

        # Create buffered frames by sliding window view, step H
        buffered = np.lib.stride_tricks.sliding_window_view(note_sig, window_shape=W)[::H]
        if buffered.ndim < 2 or buffered.shape[0] < 2:
            # Need at least two frames (original code printed a message)
            if dbg:
                print("buffered signal not minimum of 2 frames")
            continue

        # Zero-pad frames to FFT size (kept as original: padded to fft_size)
        padded_frames = np.pad(buffered, ((0, 0), (0, fft_size - W)), mode='constant')
        window = scipy.signal.windows.hann(fft_size, sym=False)
        window_sum = np.sum(window)
        buffered_windowed = padded_frames * window
        freqs = rfftfreq(fft_size, d=1.0 / sr)

        # Normalize frames (preserve original normalization approach)
        max_abs = np.max(np.abs(buffered_windowed))
        if max_abs == 0:
            # avoid divide-by-zero if silent
            continue
        buffered_windowed = buffered_windowed / max_abs

        prev_fft = rfft(buffered_windowed[0])
        note_betas = []

        for frame_idx, frame in enumerate(buffered_windowed[1:], start=1):
            curr_fft = rfft(frame)
            magnitude = np.abs(curr_fft)
            magnitude_normalized = (2.0 * magnitude) / window_sum
            magnitude_db = 20.0 * np.log10(magnitude_normalized + 1e-10)

            Threshold = -50  # kept from legacy code
            peak_indices, _ = find_peaks(magnitude_db, height=Threshold)
            if len(peak_indices) == 0:
                if dbg:
                    print("no peaks detected")
                prev_fft = curr_fft
                continue

            # estimate fundamental using phase method (original function preserved)
            k_note_freq = int(note_freq * fft_size / sr)
            fundamental = calculate_fundamental(
                k_note_freq, prev_fft, curr_fft, fft_size, H, sr, frame, window
            )

            # Validate fundamental within quarter-tone of model note frequency
            qlow = note_freq / math.pow(2.0, 1.0 / 24.0)
            qhigh = note_freq * math.pow(2.0, 1.0 / 24.0)
            if not (qlow <= fundamental <= qhigh):
                if dbg:
                    print("Fundamental not in quartertone range of noteFreq for Frame")
                prev_fft = curr_fft
                continue

            # Determine partials (preserve original cap of 25)
            n_partials = min(len(peak_indices), 20)
            partials = calculate_partials(
                fundamental,
                int(fundamental * fft_size / sr),
                prev_fft,
                curr_fft,
                fft_size,
                H,
                sr,
                beta_max,
                n_partials,
                dbg,
                frame,
                window,
                peak_indices
            )

            # Compute partial betas (skip order 0)
            partial_betas = [
                ((freq / ((order + 1) * partials[0])) ** 2 - 1) / ((order + 1) ** 2)
                for order, freq in partials.items() if order > 0
            ]

            # Keep only physically meaningful betas
            valid_betas = [b for b in partial_betas if 0 < b < beta_max]
            if valid_betas:
                # Apply IQR filtering to remove outliers
                filtered_betas = filter_outliers_iqr(valid_betas, dbg)
                if filtered_betas:
                    note_betas.append(filtered_betas)
                else:
                    if dbg:
                        print("No values left after IQR-filter")
            else:
                if dbg:
                    print("No valid partial Betas")

            prev_fft = curr_fft

            # Debug plotting (kept original plotting layout)
            if dbg:
                plt.figure(figsize=(12, 6))
                plt.plot(freqs, magnitude_db, label=f"Frame {frame_idx}")
                plt.axhline(Threshold, color='black', linestyle='--', label=f'Threshold: {Threshold:.2f} dB')
                plt.scatter(note_freq, 0, color="purple", marker="x", s=100, label="Model NoteFreq")
                plt.scatter(freqs[peak_indices], magnitude_db[peak_indices], color='orange', marker="o", label="Detected Peaks")
                predicted_partials = [fundamental * (i + 1) for i in range(1, n_partials)]
                plt.scatter(predicted_partials, [0] * len(predicted_partials), color='red', marker="s", label="Predicted Partials")
                real_partials = list(partials.values())
                plt.scatter(real_partials, [-2] * len(real_partials), color='green', marker="d", label="Real Partials")
                plt.title(f"Frequency Spectrum (Frame {frame_idx})")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.xscale("log")
                plt.grid()
                plt.legend()
                plt.show()

        # Flatten nested beta lists and attach to note object
        flat_betas = flatten_recursive(note_betas)
        if dbg:
            print(f"noteBetas: {flat_betas}")

        updated_note = dataclasses.replace(note, noteBetas=flat_betas)
        notes[idx] = updated_note

    return notes


# Note matching & model inference helpers
def fill_stringNotes(pred_notes: List, GT_matchedNotes: List, delta: float) -> List[stringNote]:
    """
    Match predicted notes with ground-truth notes and build stringNote objects.

    Args:
        pred_notes: Predicted notes from the transcription model.
        GT_matchedNotes: Ground-truth matched notes.
        delta: Maximum allowed onset time difference for matching (seconds).

    Returns:
        List of stringNote objects initialized with ground-truth string index.
    """
    string_notes: List[stringNote] = []

    for pred in pred_notes:
        for ref in GT_matchedNotes:
            if (
                abs(ref.onset - pred.onset) <= delta
                and ref.pitch == pred.pitch
                and not pred.is_drum
                and pred.program == 24
            ):
                sn = stringNote(
                    is_drum=ref.is_drum,
                    program=ref.program,
                    onset=ref.onset,
                    offset=ref.offset,
                    pitch=ref.pitch,
                    velocity=ref.velocity,
                    noteBetas=None,  # will be filled by estimate_betas_for_note
                    string_pred=None,
                    likelihood_ratio=None,
                    stringGT=ref.string_index
                )
                string_notes.append(sn)
    return string_notes


def process_file(
    sig: np.ndarray,
    sr: int,
    dbg: bool,
    model,
    audio_filepath: str,
    delta: float,
    betas_kde_dict: Dict,
    betas_kde_xvals: np.ndarray,
    GT_matchedNotes: np.ndarray,
    betas: Dict
) -> List[stringNote]:
    """
    Process an audio file: transcribe, match to ground truth, and estimate Betas.

    This implementation uses the legacy estimator `estimate_betas_for_note`.

    Args:
        sig (np.ndarray): Audio signal.
        sr (int): Sampling rate (Hz).
        dbg (bool): Enable debug plotting.
        model: Transcription model.
        audio_filepath (str): Path to the audio file (currently unused, kept for compatibility).
        delta (float): Onset tolerance for GT matching.
        betas_kde_dict (Dict): Precomputed KDE dictionary (unused, kept for compatibility).
        betas_kde_xvals (np.ndarray): Grid for KDE evaluation (unused).
        GT_matchedNotes (np.ndarray): Ground-truth matched notes.
        betas (Dict): Beta distribution dictionary loaded from JSON (kept for compatibility).

    Returns:
        List[stringNote]: List of stringNote objects with estimated noteBetas.
    """
    # Step 1: Transcribe audio into note events
    audio_tensor = torch.from_numpy(sig).float()
    audio_info = prepare_media(audio_tensor, sr)
    print("Audio info:", audio_info)
    pred_notes = transcribe_notes(model, audio_info, audio_tensor, sr)

    # Step 2: Match predictions to ground truth
    string_notes = fill_stringNotes(pred_notes, GT_matchedNotes, delta)

    # Step 3: Estimate Beta distribution for each note
    string_notes = estimate_betas_for_note(sig, sr, string_notes, dbg)

    return string_notes








def main(argv=None):
    """
    Loads model + betas, computes KDEs for the betas JSON,
    runs through the audio folder (annotation matching) and processes up to max_files.

    CLI flags:
        --debug: toggle debug plotting (default False)
        --max-files: limit processed files (default 5)
    """
    parser = argparse.ArgumentParser(description="Estimate per-note inharmonicity betas and identify strings.")
    parser.add_argument("--debug", action="store_true", help="Enable debug plotting and messages.")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum number of files to process (default 5).")
    args = parser.parse_args(argv)

    dbg = bool(args.debug)
    max_files = int(args.max_files)

    model_name = "YPTF+Single (noPS)"
    print(f"Running evaluation for model: {model_name}")

    # Map model name to checkpoint args (kept original choices)
    project = '2024'
    precision = 'bf16-mixed'

    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        ckpt_args = [checkpoint, '-p', project, '-pr', precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        ckpt_args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
                     '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        ckpt_args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
                     '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
                     '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        ckpt_args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                     '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                     '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                     '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        ckpt_args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                     '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                     '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                     '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load model checkpoint (kept unchanged)
    model = load_model_checkpoint(args=ckpt_args)

    # Load betas from JSON and compute KDE dictionary for strings 1..6 (Saite_1 .. Saite_6)
    betas_json_path = '../content/Betas/betas.json'
    try:
        betas = load_betas_from_json(betas_json_path)
    except FileNotFoundError:
        print(f"Beta JSON not found: {betas_json_path}. Continuing with empty betas dict.")
        betas = {}

    string_keys = [f"Saite_{i + 1}" for i in range(6)]
    betas_kde_x_vals = np.linspace(0.0, 0.001, 200)
    betas_kde_dict = {}
    for key in string_keys:
        raw_vals = np.array([v for v in betas.get(key, []) if 0.0 <= v <= 0.001])
        betas_kde_dict[key] = compute_kde_normalized(raw_vals, betas_kde_x_vals)

    # Iteration over audio files + processing
    audio_directory = '../../data/guitarset_yourmt3_16k/audio_mono-mic/'
    annotation_directory = '../../data/guitarset_yourmt3_16k/annotation/'

    all_string_notes: List[stringNote] = []
    filtered_string_notes: List[stringNote] = []

    processed_count = 0
    for file in sorted(os.listdir(audio_directory)):
        if "pshift" in file:
            # original script excluded pshifted files
            continue

        audio_filepath = os.path.join(audio_directory, file)
        annotation_filename = get_annotation_filename(file)
        annotation_filepath = os.path.join(annotation_directory, annotation_filename)

        if not os.path.exists(annotation_filepath):
            if dbg:
                print(f"Annotation not found for {file}: {annotation_filepath}")
            continue

        # load GT matched notes
        try:
            GT_matchedNotes = extract_GT(annotation_filename, annotation_directory)
        except FileNotFoundError as e:
            if dbg:
                print(str(e))
            continue

        # read audio file
        sig, sr = sf.read(audio_filepath)
        if sig.ndim > 1:
            # if stereo, make mono by averaging channels (preserve simple approach)
            sig = np.mean(sig, axis=1)
        # normalize
        max_abs = np.max(np.abs(sig)) if np.max(np.abs(sig)) != 0 else 1.0
        sig = sig / max_abs

        # process file
        delta = 0.05  # 50 ms tolerance (kept original default)
        stringNotes = process_file(sig, sr, dbg, model, audio_filepath, delta,
                                   betas_kde_dict, betas_kde_x_vals, GT_matchedNotes, betas)

        # Save per-file stringNotes
        ann_base = os.path.splitext(annotation_filename)[0].replace("_notes", "")
        save_dir = annotation_directory
        save_path = os.path.join(save_dir, f"{ann_base}_stringNotes.npy")
        np.save(save_path, np.array(stringNotes, dtype=object))

        all_string_notes.extend(stringNotes)
        processed_count += 1
        print("Analysed File-Number:", processed_count)

        # if processed_count >= max_files:
        #     break

    # Apply string classification method (the original script used a particular method)
    all_string_notes = wasserstein_freq_semi_empirical(all_string_notes, betas)

    # Filter out notes with no betas (original behavior)
    filtered_string_notes = [n for n in all_string_notes if n.noteBetas is not None]

    # Analyze results
    analyze_string_predictions(filtered_string_notes)

    # Save aggregated results
    save_dir = annotation_directory
    np.save(os.path.join(save_dir, "000all_stringNotes.npy"), np.array(all_string_notes, dtype=object))
    np.save(os.path.join(save_dir, "000all_stringNotes_filtered.npy"), np.array(filtered_string_notes, dtype=object))

    print("Done.")


if __name__ == "__main__":
    main()