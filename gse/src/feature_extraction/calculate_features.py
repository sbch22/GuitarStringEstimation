"""
Feature calculation pipeline for guitar string tracks.

Processes pickled track files: for each note, runs inharmonic partial
tracking, extracts spectral and partial-based features, and saves
results back to the track file.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle

import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt
from scipy import stats

from gse.src.utils.FeatureNote_dataclass import FilterReason, Features

from gse.src.feature_extraction.inharmonic_partial_tracking import (
    kde_mode,
    inharmonic_partial_tracker,
    note_audio_preprocess,
)
from gse.src.feature_extraction.feature_functions import (
    filter_betas,
    spectral_centroid_feature,
    relative_amplitude_deviations,
    relative_freq_deviations,
)


# ── Per-note processing ───────────────────────────────────────────────────────
def process_note(
    note,
    note_signal,
    audio_type,
    beta_min,
    beta_max,
    sr,
    W,
    H,
    threshold,
    plot,
):
    """Run partial tracking and feature extraction for a single note."""
    # --- init features ---
    if not isinstance(note.features, dict):
        note.features = {}
    if audio_type not in note.features:
        note.features[audio_type] = Features()

    # --- audio slicing ---
    onset_sample = int(note.attributes.onset * sr)
    offset_sample = int(note.attributes.offset * sr)
    audio_data = note_signal.time
    string_idx = note.attributes.string_index

    if audio_type == "hex_debleeded":
        note_audio = audio_data[string_idx, onset_sample:offset_sample]
    else:
        if audio_data.ndim == 1:
            note_audio = audio_data[onset_sample:offset_sample]
        else:
            note_audio = audio_data[0, onset_sample:offset_sample]

    if note_audio is None:
        note.invalidate(FilterReason.NO_NOTE_AUDIO, step="find_partials")
        return

    # --- preprocessing ---
    harmonic_audio = note_audio
    if len(harmonic_audio) < W:
        note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
        return

    N_fft = 16 * W  # zero-padding for low-frequency resolution
    preprocessed_audio, window = note_audio_preprocess(harmonic_audio, W, H, N_fft)

    if preprocessed_audio.ndim < 2:
        note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
        return

    fft_frames = np.fft.rfft(preprocessed_audio, axis=1)

    # --- partial tracking ---
    partials, betas, beta, f0 = inharmonic_partial_tracker(
        fft_frames=fft_frames,
        f0=note.attributes.pitch,
        beta=beta_min,
        n_iter=50,
        sr=sr,
        N_fft=N_fft,
        H=H,
        threshold=threshold,
        note_name=str(note.attributes.pitch),
        plot=plot,
        beta_max=beta_max,
    )

    note.partials[audio_type] = partials

    if partials is None:
        note.invalidate(FilterReason.NO_PARTIALS_FOUND, step="find_partials")
        return

    # --- beta post-processing ---
    betas = filter_betas(betas, beta_max)
    betas = np.array(betas)

    beta = kde_mode(betas)
    if np.isnan(beta):
        beta = np.nanmedian(betas)

    # --- optional spectrogram plot ---
    if plot:
        _plot_spectrogram_with_partials(fft_frames, partials, sr, N_fft, H, beta, note)

    # --- feature computation ---
    feat = note.features[audio_type]
    feat.f0 = note.attributes.pitch
    feat.beta = beta

    if np.any(np.isfinite(betas)):
        feat.betas_measures = np.array([
            np.nanmean(betas),
            np.nanmedian(betas),
            np.nanstd(betas),
            np.nanvar(betas),
            np.nanmin(betas),
            np.nanmax(betas),
            stats.skew(betas, nan_policy="omit"),
            stats.kurtosis(betas, nan_policy="omit"),
            kde_mode(betas),
        ])
    else:
        feat.betas_measures = np.full(9, np.nan)

    feat.spectral_centroid = spectral_centroid_feature(harmonic_audio, W, H, sr)
    feat.rel_partial_amplitudes, feat.amp_decay_coefficients = (
        relative_amplitude_deviations(partials)
    )
    feat.rel_freq_deviations = relative_freq_deviations(partials, beta)

    valid_flags = np.isfinite(partials.amplitudes)
    feat.valid_partials = valid_flags.mean(axis=0)

    feat.fill_feature_vector()


# ── Plotting helper ───────────────────────────────────────────────────────────
def _plot_spectrogram_with_partials(fft_frames, partials, sr, N_fft, H, beta, note):
    """Overlay tracked partials on a normalised spectrogram."""
    n_frames, n_bins = fft_frames.shape
    freq_axis = np.arange(n_bins) * sr / N_fft

    fft_mag = np.abs(fft_frames)
    fft_norm = fft_mag / (fft_mag.max() + 1e-12)
    fft_norm_db = 20 * np.log10(fft_norm)

    times_frames = np.arange(n_frames) * (H / sr)

    fig, ax = plt.subplots(figsize=(12, 8))
    pcm = ax.pcolormesh(
        times_frames, freq_axis, fft_norm_db.T,
        shading="auto", cmap="magma",
        vmin=fft_norm_db.max() - 80,
        vmax=fft_norm_db.max(),
    )

    partial_freqs = partials.frequencies
    for p in range(partial_freqs.shape[1]):
        ax.plot(times_frames, partial_freqs[:, p], color="lime", alpha=0.8, linewidth=2)

    ax.set_ylim(80, min(2000, sr / 2))
    ax.set_title(
        f"{note.attributes.pitch} Hz | β={beta:.2e} | String: {note.attributes.string_index}"
    )
    plt.colorbar(pcm, ax=ax)
    plt.tight_layout()
    plt.show()


# ── Per-file processing ──────────────────────────────────────────────────────
def process_single_file(args):
    """Load a track file, compute features for all notes, and save."""
    filepath, beta_min, beta_max, plot, threshold, W, H, audio_types = args

    print(f"Calculating Features {filepath}")

    with open(filepath, "rb") as f:
        track = pickle.load(f)

    for audio_type in audio_types:
        audio_filepath = track.audio_paths[audio_type]
        note_signal = pf.io.read_audio(audio_filepath)
        note_signal = pf.dsp.normalize(note_signal)
        sr = note_signal.sampling_rate

        for note in track.valid_notes:
            process_note(
                note=note,
                note_signal=note_signal,
                audio_type=audio_type,
                beta_min=beta_min,
                beta_max=beta_max,
                sr=sr,
                W=W,
                H=H,
                threshold=threshold,
                plot=plot,
            )

    track.save(filepath)
    print(f"Saved {filepath}")

    return f"Success! Feature Vector (raw) calculated for: {os.path.basename(filepath)}"


def main(config):
    """Read config, collect track files, and run feature extraction in parallel."""
    W = config.getint('params', 'W')
    H = config.getint('params', 'H')
    beta0_min = config.getfloat('params', 'beta0_min')
    beta0_max = config.getfloat('params', 'beta0_max')
    threshold = config.getint('params', 'threshold')
    plot = config.getboolean('params', 'plot')
    audio_types_raw = config.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]
    track_directory = config.get('paths', 'track_directory')

    beta_min = beta0_min
    beta_max = beta0_max * 2 ** (20 / 6)  # 20th fret as upper boundary

    print(W, H, beta_min, beta_max, threshold)
    print(track_directory)

    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if filename.endswith(".pkl")
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    args_list = [
        (fp, beta_min, beta_max, plot, threshold, W, H, audio_types)
        for fp in filepaths
    ]

    num_processes = mp.cpu_count() - 1
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, args_list)

    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")
