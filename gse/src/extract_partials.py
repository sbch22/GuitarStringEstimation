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
from scipy.signal import medfilt


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
        # asymmetrisches Fenster: nach unten breiter
        b_lo = max(b_f0 - 5, 0) if prev_bin is None else max(prev_bin - 2, 0)
        b_hi = min(b_f0 + 3, bin_nyquist) if prev_bin is None else min(prev_bin + 2, bin_nyquist)

        amp_region = inst_amp[t, b_lo:b_hi+1]
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
    max_jump_hz = 20  # maximale Sprungweite zwischen Frames (zeitkontinuität)

    for t in range(n_frames):
        if np.isnan(f0_frame[t]):
            continue

        f0_t = f0_frame[t]

        for p_idx, h in enumerate(harmonic_orders):
            # erwartete Partialfrequenz
            f_expected = h * f0_t
            f_min = f_expected * np.sqrt(1 - beta_max * h**2)
            f_max = f_expected * np.sqrt(1 + beta_max * h**2)

            # bin Bereich
            b_exp = int(f_expected * W / sr)
            b_lo = max(int(f_min * W / sr), 0)
            b_hi = min(int(f_max * W / sr), bin_nyquist)

            # nur minimal 1 bin
            if b_hi < b_lo:
                b_hi = b_lo + 1

            amp_region = inst_amp[t, b_lo:b_hi+1]
            freq_region = inst_freq[t, b_lo:b_hi+1]

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
            amp_db = 20 * np.log10(amp + 1e-12)
            if amp_db <= threshold:
                continue

            # Ausgeben
            partial_freqs[t, p_idx] = freq
            partial_amps[t, p_idx] = amp_db
            partial_bins[t, p_idx] = b_lo + idx

    return partial_freqs, partial_amps, partial_bins



def process_track_extract_partials(track, W, H, beta_max,  n_partials, plot):
    string_hex_audio = track.audio.hex_debleeded
    sr = string_hex_audio.sampling_rate



    # 6 x n_samples Matrix
    strings_audio_matrix = string_hex_audio.time

    for note in track.notes:
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
        process_track_extract_partials(track, W, H, beta_max, n_partials=25, plot=False)

        # Save track
        track.save(filepath)

        filename = os.path.basename(filepath)
        return f"Success: {filename}"

    except Exception as e:
        return f"Error processing {filepath}: {str(e)}"


def main():
    track_directory = '../noteData/GuitarSet/train/dev/'

    # Parameters
    W = 1600
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