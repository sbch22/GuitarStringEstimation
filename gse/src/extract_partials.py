import os
import sys

sys.path.append(os.path.abspath(''))

import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from utils.FeatureNote_dataclass import Partials
import multiprocessing as mp
import os
import pickle


def instantaneous_frequency(frames, W, H, sr, window):
    """
    Calculates the instantaneous frequency of all passed frames for all possible bins.

    Args:
        frames: np.ndarray collecting all frames of raw audio data
        W: STFT-Window size
        H: Hop sizze
        sr: sampling_rate
        window: Hann or other type window with width W, for normalization

    Returns:

    """
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

    # Partial orders: partial 0 = fundamental
    partial_orders = np.arange(0, n_partials)
    harmonic_orders = partial_orders + 1

    # limit bin to nyquist freq
    bin_nyquist = W // 2

    # Expected partial frequency for each partial
    f_expected = harmonic_orders * f0
    f_min = f_expected * np.sqrt(1 - beta_max * harmonic_orders**2)
    f_max = f_expected * np.sqrt(1 + beta_max * harmonic_orders**2)

    # Convert frequency range to FFT bin range
    bin_expected = (f_expected * W / sr).astype(int)
    bin_max = (f_max * W / sr).astype(int)

    # Loop over partials (vectorized over frames inside)
    for p_idx, (b_exp, b_max_allowed) in enumerate(zip(bin_expected, bin_max)):

        # wenn
        if p_idx == 0:
            b_exp = b_exp - 1
        # clamp bins
        b_min = max(b_exp, 0)
        b_max_allowed = min(b_max_allowed, bin_nyquist)

        if b_min > b_max_allowed:
            continue

        # region for all frames
        amp_region = inst_amp[:, b_min:b_max_allowed+1]             # shape (frames, region_bins)
        freq_region = inst_freq[:, b_min:b_max_allowed+1]           # same shape

        # max amplitude in allowed region (per frame)
        local_argmax = np.argmax(amp_region, axis=1)                # (frames,)
        max_amp = amp_region[np.arange(n_frames), local_argmax]     # (frames,)
        max_bin = b_min + local_argmax                              # (frames,)
        max_freq = freq_region[np.arange(n_frames), local_argmax]   # (frames,)

        # convert amplitude to dB
        max_amp_db = 20 * np.log10(max_amp + 1e-12)

        # accept only amplitudes over threshold
        valid = max_amp_db > threshold


        # fill into outputs
        partial_freqs[valid, p_idx] = max_freq[valid]
        partial_amps[valid, p_idx]  = max_amp_db[valid]
        partial_bins[valid, p_idx]  = max_bin[valid]

    return partial_freqs, partial_amps, partial_bins



def process_track_extract_partials(track, W, H, beta_max,  n_partials, plot):
    string_hex_audio = track.audio.hex_debleeded
    sr = string_hex_audio.sampling_rate

    good_notes_before = [n for n in track.notes if n.origin == "model" and n.match == True]

    # clean up mis-assigned string ntoes
    ratio_deleted_noted = track.match_notes_between_strings(string_hex_audio, 0.05, track.notes)

    good_notes_after = [n for n in track.notes if n.origin == "model" and n.match == True]

    # 6 x n_samples Matrix
    strings_audio_matrix = string_hex_audio.time

    for note in good_notes_after:
        # extract note audio from
        onset_sample = int(note.attributes.onset * sr)
        offset_sample = int(note.attributes.offset * sr)
        # valid notes
        if  note.match is not True or note.attributes.midi_note is None or offset_sample - onset_sample < W :
            continue # use only model notes

        if note.origin != "model":
            continue

        noteMIDI = round(note.attributes.midi_note)
        note.attributes.pitch = (440 / 32) * (2 ** ((noteMIDI - 9) / 12))

        k_f0 = int(note.attributes.pitch * W / sr)

        note_audio = strings_audio_matrix[note.attributes.string_index, onset_sample:offset_sample]
        note_audio /= np.max(np.abs(note_audio))

        # skip notes too short
        if note_audio.size < W:
            continue

        # buffer signal
        note_audio = np.lib.stride_tricks.sliding_window_view(note_audio, window_shape=W)[::H]
        if note_audio.ndim < 2:
            print("Not enough frames for analysis")
            continue

        note_audio = np.pad(note_audio, ((0, 0), (0, W - note_audio.shape[1])), mode='constant')
        window = scipy.signal.windows.hann(W, sym=False)
        note_audio = note_audio * window

        inst_freq, inst_amp = instantaneous_frequency(note_audio, W, H, sr, window)

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
            threshold = -60,
        )

        # Zeitachse
        t_frames = np.arange(partial_freqs.shape[0]) * (H / sr)
        if plot:
            # --- Spectrogram with partials overlay ---
            fft_mag = np.abs(np.fft.rfft(note_audio, axis=1))
            fft_mag_db = 20 * np.log10(fft_mag + 1e-12)

            # Drop first FFT frame to match inst_freq length
            fft_mag_db_if = fft_mag_db[1:]
            times_if = np.arange(fft_mag_db_if.shape[0]) * (H / sr)

            freqs = np.fft.rfftfreq(W, 1 / sr)
            times = np.arange(fft_mag_db.shape[0]) * (H / sr)

            plt.figure(figsize=(14, 6))
            plt.imshow(
                fft_mag_db_if.T,
                origin="lower",
                aspect="auto",
                extent=[times_if[0], times_if[-1], freqs[0], freqs[-1]],
                cmap="magma",
                vmin=-100,
                vmax=0,
            )

            # Overlay partials (now aligned)
            for p in range(partial_freqs.shape[1]):
                plt.plot(times_if, partial_freqs[:, p], "c", linewidth=1.5)

            plt.colorbar(label="Magnitude (dB)")
            plt.title(f"Spectrogram with Extracted Partials – {note.attributes.pitch:.2f} Hz")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0, 5000)
            plt.tight_layout()
            plt.show()

            # --- 1. Line Plot ---
            plt.figure(figsize=(14, 7))
            for p_idx in range(partial_freqs.shape[1]):
                plt.plot(t_frames, partial_freqs[:, p_idx], label=f"P{p_idx}", linewidth=1.5)
            plt.title(f"Instantaneous Partial Frequencies – Note: {note.attributes.pitch:0.2f} Hz")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=4, fontsize=9)
            plt.tight_layout()
            plt.show()

        # save into Partials object
        note.partials = Partials(
            frametimes=t_frames,
            frequencies=partial_freqs,
            amplitudes=partial_amps,
        )




def process_single_file(args):
    """Worker function to process a single file"""
    filepath, W, H, beta_max = args

    try:
        # Load track
        with open(filepath, "rb") as f:
            track = pickle.load(f)

        # Process track
        process_track_extract_partials(track, W, H, beta_max, n_partials=25, plot=True)

        # Save track
        track.save(filepath)

        filename = os.path.basename(filepath)
        return f"Success: {filename}"

    except Exception as e:
        return f"Error processing {filepath}: {str(e)}"


def main():
    track_directory = '../noteData/GuitarSet/train/dev/'

    # Parameters
    W = 2048
    H = int(W / 8)
    beta_max = 1e-4

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    # Prepare arguments for each file
    args_list = [(fp, W, H, beta_max) for fp in filepaths]

    # Create pool and process files
    num_processes = mp.cpu_count() - 1  # Leave one core free
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, args_list)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()