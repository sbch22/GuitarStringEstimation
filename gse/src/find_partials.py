import os
import sys

from librosa.feature import spectral_centroid

sys.path.append(os.path.abspath(''))

import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from gse.src.utils.FeatureNote_dataclass import Partials
import multiprocessing as mp
import os
import pickle
from scipy.signal import medfilt
import librosa as lb
import sounddevice as sd
from configparser import ConfigParser
from multiprocessing import Pool, cpu_count
import pyfar as pf

from collections import defaultdict
from utils.FeatureNote_dataclass import FilterReason, Features



def note_audio_preprocess(audio, W, H):
    """
    Prerocesses Audio:
    - Pad Audio
    - Buffering
    - Windowing
    Args:
        audio: note audio
        W: W
        H: H

    Returns:
        audio_preprocessed: preprocessed note audio`

    """
    # extract harmonic part of note audi -> cut out other onsets & time window the rest
    # harmonic_audio, intra_onsets = extract_harmonic_note_audio(audio, W, H, sr, plot)

    # Pad the audio so last window is included
    audio_buffered = np.pad(audio, (0, W), mode="constant")

    # buffer signal
    audio_buffered = np.lib.stride_tricks.sliding_window_view(audio_buffered, window_shape=W)[::H]

    # Apply Hann-Window on each frame
    window = scipy.signal.windows.hann(W, sym=False)
    audio_preprocessed = audio_buffered * window

    return audio_preprocessed, window


def instantaneous_frequency(frames, W, H, sr, window):
    # FFT
    fft_frames = np.fft.rfft(frames, axis=1)  # (n_frames, n_bins)
    phase = np.angle(fft_frames)
    mag = np.abs(fft_frames)
    n_bins = fft_frames.shape[1]

    # phase difference
    prev_phase = phase[:-1, :]
    curr_phase = phase[1:, :]

    k = np.arange(n_bins)                     # bins
    omega_k = 2 * np.pi * k / W

    epsilon = 1e-10
    delta_phi = (
        omega_k * H
        + np.mod(curr_phase - prev_phase - omega_k * H + np.pi + epsilon, 2*np.pi)
        - np.pi
    )

    # Instantaneous Frequency
    inst_freq = (delta_phi / (2*np.pi*H)) * sr    # (n_frames-1, n_bins)
    inst_amp  = 2 * mag[1:, :] / np.sum(window)   # same shape

    return inst_freq, inst_amp


def partial_picker(inst_freq, inst_amp, f0, k_f0, beta_max, n_partials, sr, W, threshold):
    n_frames, n_bins = inst_freq.shape

    # Outputs
    partial_freqs = np.full((n_frames, n_partials), np.nan)
    partial_amps  = np.full((n_frames, n_partials), np.nan)
    partial_bins  = np.full((n_frames, n_partials), -1, dtype=int)

    # Partial orders
    partial_orders = np.arange(0, n_partials)
    harmonic_orders = partial_orders + 1

    # Nyquist
    bin_nyquist = W // 2

    # ----- 1. f0 pro Frame suchen -----
    f0_frame = np.full(n_frames, np.nan)
    prev_bin = None

    for t in range(n_frames):
        b_f0 = int(f0 * W / sr)

        # wide search range -> smaller search range
        b_lo = max(b_f0 - 2, 0) if prev_bin is None else max(prev_bin - 1, 0)
        b_hi = min(b_f0 + 2, bin_nyquist) if prev_bin is None else min(prev_bin + 1, bin_nyquist)

        amp_region = inst_amp[t, b_lo-1:b_hi+1]
        if amp_region.size == 0:
            continue

        idx0 = np.argmax(amp_region)
        f0_t = inst_freq[t, b_lo + idx0]
        f0_frame[t] = f0_t
        prev_bin = b_lo + idx0  # 6. letzes Fenster als Startwert

    # ----- 2. f0 glätten -----
    # Medianfilter über 5 Frames
    f0_frame = medfilt(f0_frame, kernel_size=5)

    # ----- 3. Partials pro Frame bestimmen -----
    max_jump_hz = 50  # maximale Sprungweite zwischen Frames (zeitkontinuität)

    for t in range(n_frames):
        if np.isnan(f0_frame[t]):
            continue

        f0_t = f0_frame[t]
        tol = 1e-5

        for p_idx, h in enumerate(harmonic_orders):
            # erwartete Partialfrequenz
            f_min = h * f0_t * (1 - tol)
            f_max = h * f0_t * np.sqrt(1 + beta_max * h ** 2)

            # bin Bereich
            b_lo = max(int(f_min * W / sr) - 3, 0)
            b_hi = min(int(f_max * W / sr) + 3, bin_nyquist)

            # nur minimal 1 bin
            if b_hi < b_lo:
                b_hi = b_lo + 1
                b_lo = b_lo - 1

            amp_region = inst_amp[t, b_lo:b_hi]
            freq_region = inst_freq[t, b_lo:b_hi]

            if amp_region.size == 0:
                continue

            idx = np.argmax(amp_region)
            amp = amp_region[idx]
            freq = freq_region[idx]

            # 4. Zeitkontinuität prüfen
            if t > 0 and not np.isnan(partial_freqs[t-1, p_idx]):
                if abs(freq - partial_freqs[t-1, p_idx]) > max_jump_hz:
                    freq = partial_freqs[t-1, p_idx]
                    amp = partial_amps[t-1, p_idx]
                    idx = partial_bins[t-1, p_idx] - b_lo  # passt ungefähr

            # Threshold prüfen
            if np.isnan(amp):
                amp_db = np.nan
            elif amp > 0:
                amp_db = 20 * np.log10(amp)
                if amp_db <= threshold:
                    continue
            else:
                continue  # amp == 0, nothing to keep

            # Ausgeben
            partial_freqs[t, p_idx] = freq
            partial_amps[t, p_idx] = amp_db
            partial_bins[t, p_idx] = b_lo + idx

    return partial_freqs, partial_amps, partial_bins


def extract_harmonic_note_audio(note_audio, W, H, sr, plot):
    note_audio /= np.max(np.abs(note_audio))

    # skip notes too short
    if note_audio.size < W:
        return None, None

    # Onset detection in note
    H_onset = int(H / 4)
    kwargs = {
        'pre_max': 2,
        'post_max': 2,
        'pre_avg': 1,
        'post_avg': 1,
        'delta': 0.2,  # größerer Wert = strengere Peaks
        'wait': H_onset,  # z.B. ~1ms bei sr=44k
    }

    intra_onsets = lb.onset.onset_detect(
        y=note_audio,
        sr=sr,
        hop_length=H_onset,
        units='samples',
        backtrack=False,
        sparse=True,
        **kwargs
    )

    # default harmonic region
    note_len = len(note_audio)
    harmonic_start = 0  # always original onset
    harmonic_end = note_len  # default: keep full note

    # first intra onset in second half of note
    if len(intra_onsets) > 0:
        second_half_onsets = intra_onsets[intra_onsets >= note_len / 2]

        if len(second_half_onsets) > 0:
            harmonic_end = int(second_half_onsets[0])

    harmonic_audio = note_audio[harmonic_start:harmonic_end]

    # safety check
    if len(harmonic_audio) < W:
        return None, None

    # --- harmonic slice ---
    harmonic_audio_raw = note_audio[harmonic_start:harmonic_end]

    # window
    harmonic_window = scipy.signal.windows.hann(
        len(harmonic_audio_raw), sym=False
    )
    harmonic_audio_win = harmonic_audio_raw * harmonic_window
    harmonic_audio = harmonic_audio_win / len(harmonic_audio_win)

    if plot:
        plt.figure(figsize=(12, 3))

        # 1) raw note (ungefenstert)
        plt.plot(
            np.arange(len(note_audio)),
            note_audio,
            color="black",
            alpha=0.5,
            linewidth=1.5,
            label="note (raw)",
        )

        # 2) harmonic part (windowed, korrekt positioniert)
        plt.plot(
            np.arange(harmonic_start, harmonic_end),
            harmonic_audio_win,
            color="blue",
            linewidth=2,
            label="harmonic (hann-windowed)",
        )

        plt.legend()
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Raw note + windowed harmonic part")
        plt.tight_layout()
        plt.show()

        # play harmonic part of note
        sd.play(harmonic_audio, sr)
        sd.wait()

    return harmonic_audio_win, intra_onsets


def process_track_extract_partials(track, W, H, beta_max, audio_type, n_partials, plot, threshold):
    filepath = track.audio_paths[str(audio_type)]
    audio = pf.io.read_audio(filepath)
    sr = audio.sampling_rate

    # 6 x n_samples Matrix
    audio_data = audio.time

    for note in track.valid_notes:
        # extract note audio from
        onset_sample = int(note.attributes.onset * sr)
        offset_sample = int(note.attributes.offset * sr)

        k_f0 = int(note.attributes.pitch * W / sr)

        note_audio = audio_data[onset_sample:offset_sample]

        if note_audio is None:
            note.invalidate(FilterReason.NO_HARMONIC_AUDIO, step="find_partials")
            continue

        preprocessed_audio, window = note_audio_preprocess(note_audio, W, H)

        if preprocessed_audio.ndim < 2:
            note.invalidate(FilterReason.HARMONIC_AUDIO_TOO_SHORT, step="find_partials")
            continue

        # Or guard before assignment in process_track_extract_partials:
        if note.features is None:
            note.features = Features()

        # calculate accurate freq & amplitude for all possible bins
        inst_freq, inst_amp = instantaneous_frequency(preprocessed_audio, W, H, sr, window)

        # pick best partials
        partial_freqs, partial_amps, partial_bins = partial_picker(
            inst_freq,
            inst_amp,
            f0=note.attributes.pitch,
            k_f0=k_f0,
            beta_max=beta_max,
            n_partials=n_partials,
            sr=sr,
            W=W,
            threshold = threshold,
        )


        # Zeitachse
        t_frames = np.arange(partial_freqs.shape[0]) * (H / sr)

        if plot:
            # --- Spectrogram with partials overlay ---
            fft_mag = np.abs(np.fft.rfft(preprocessed_audio, axis=1))
            fft_mag_db = 20 * np.log10(fft_mag + 1e-12)

            # Drop first FFT frame to match inst_freq length
            fft_mag_db_if = fft_mag_db[1:]
            times_if = np.arange(fft_mag_db_if.shape[0]) * (H / sr)

            freqs = np.fft.rfftfreq(W, 1 / sr)

            plt.figure(figsize=(14, 12))

            pcm = plt.pcolormesh(
                times_if,
                freqs,
                fft_mag_db_if.T,
                shading="auto",
                cmap="magma",
                vmin=threshold,
                vmax=2,
            )

            # Overlay partials
            for p in range(partial_freqs.shape[1]):
                plt.plot(times_if, partial_freqs[:, p], linewidth=2.5, color='g')

            # Onsets als vertikale schwarze Linien
            for onset in intra_onsets:
                plt.axvline(
                    x=onset / sr,  # 👈 FIX
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )

            # plt.yscale("log")
            plt.ylim(80, 8000)
            plt.colorbar(pcm, label="Magnitude (dB)")
            plt.title(
                f"Spectrogram with Extracted Partials – "
                f"{note.attributes.pitch:.2f} Hz, String: {note.attributes.string_index}"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            # plt.legend(ncol=4, fontsize=9)
            plt.tight_layout()
            plt.show()

        # save into Partials object
        note.partials = Partials(
            frametimes=t_frames,
            frequencies=partial_freqs,
            amplitudes=partial_amps,
        )

        # Errors in Filtering
        if note.partials == None:
            note.invalidate(FilterReason.NO_PARTIALS_FOUND, step="find_partials")


def process_single_file(filepath, W, H, beta_max, plot, threshold, audio_type):
    """Worker function to process a single file"""
    # load track
    try:
        with open(filepath, "rb") as f:
            track = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except EOFError:
        print(f"File corrupted/empty: {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        return None

    # Process track
    process_track_extract_partials(track, W, H, beta_max, audio_type, n_partials=25, plot=plot, threshold = threshold)
    track.save(filepath)


def process_file_wrapper(args):
    filepath, idx, total, W, H, beta_max, plot, threshold, audio_type = args
    print(f"\n[{idx}/{total}] Processing {filepath}")
    return process_single_file(filepath, W, H, beta_max, plot, threshold, audio_type)


def main(config):
    track_directory = config.get('paths', 'track_directory')
    W = config.getint('params', 'W')
    H = config.getint('params', 'H')
    beta_max = config.getfloat('params', 'beta_max')
    threshold = config.getint('params', 'threshold')
    plot = config.getboolean('params', 'plot')
    audio_type = config.get('paths', 'audio_type')

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    total = len(filepaths)
    args_list = [
        (filepath, idx, total, W, H, beta_max, plot, threshold, audio_type)
        for idx, filepath in enumerate(filepaths, 1)
    ]

    num_workers = cpu_count()
    print(f"Processing {total} files using {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        pool.map(process_file_wrapper, args_list)
    print(f"\nDone.")

if __name__ == "__main__":
    config = ConfigParser()
    config.read('config_train.ini')

    main(config)