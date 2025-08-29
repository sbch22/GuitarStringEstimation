import os
import sys

sys.path.append(os.path.abspath(''))


import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import json
from collections import namedtuple
import math
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.signal import find_peaks

from scipy import interpolate




def estimate_frequency_from_phase(prev_phase, current_phase, k, W, H, sr):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k
    # Berechne die Phasendifferenz und verhindere Phasenwrapping
    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi
    # Frequenzberechnung
    est_freq = delta_phi / (2 * np.pi * H) * sr
    return est_freq


def estimate_frequency_and_amplitude(prev_phase, current_phase, k, W, H, sr, frame, window):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k

    # Berechne die Phasendifferenz und verhindere Phasenwrapping
    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi

    # Frequenzberechnung
    est_freq = delta_phi / (2 * np.pi * H) * sr

    # Amplitudenschätzung über DFT-Koeffizient
    n = np.arange(W)
    dft_coeff = np.dot(frame * window, np.exp(1j * 2 * np.pi * est_freq / sr * n))
    est_amplitude = 2 * np.abs(dft_coeff) / np.sum(window)

    return est_freq, est_amplitude


def calculate_partials(fundamental, k_fundamental, prev_NOTESIG, NOTESIG, fft_size, H, sr, beta_max,
                       n_partials, dbg, frame, window, peak_indices):
    partials = {0: fundamental}

    for i in range(1, n_partials):
        predicted_partial = (i + 1) * fundamental
        maxBeta_partial = predicted_partial * math.sqrt(1 + beta_max * (i + 1) ** 2)

        k = int((i + 1) * k_fundamental)
        if k >= fft_size // 2:  # Sicherstellen, dass k im gültigen Bereich bleibt
            break

        best_partial = None
        best_amplitude = -float('inf')
        best_offset = None

        # 🔍 **Peak innerhalb des Frequenzbereichs suchen**
        peak_candidates = [p for p in peak_indices if predicted_partial <= p * sr / fft_size <= maxBeta_partial]

        if peak_candidates:
            # Wähle den Peak mit der höchsten Amplitude als Basis
            peak_bin = max(peak_candidates, key=lambda p: np.abs(NOTESIG[p]))

            # 🔍 **Untersuche nur die Umgebung (-2 bis +8 Bins)**
            for offset in range(-2, 9):
                k_test = peak_bin + offset
                if k_test < 0 or k_test >= fft_size // 2:
                    continue  # Überspringe ungültige Werte

                partial, partial_amplitude = estimate_frequency_and_amplitude(
                    np.angle(prev_NOTESIG[k_test]), np.angle(NOTESIG[k_test]), k_test, fft_size, H, sr, frame, window)

                partial_AMP = 20 * np.log10(np.abs(partial_amplitude))

                if partial_AMP > -40 and partial_AMP > best_amplitude:
                    best_partial = partial
                    best_amplitude = partial_AMP
                    best_offset = k_test - peak_bin  # Offset relativ zum Peak-Bin

        # Falls ein valider Partial gefunden wurde, speichere ihn
        if best_partial is not None:
            partials[i] = best_partial
            if dbg:
                print(f"Bin-Offset {best_offset} used for partial {i} with amplitude {best_amplitude} dB")

    if dbg:
        print(f"Length Partials: {len(partials)}")

    return partials



def calculate_fundamental(k_fundamental_guess, prev_NOTESIG, NOTESIG, fft_size, H, sr, frame, window):
    best_partial = None
    best_amplitude = -float('inf')

    for offset in range(-2, 2):  # Durchsuche k-3 bis k+20
        k_test = k_fundamental_guess + offset
        if k_test < 0 or k_test >= fft_size // 2:
            continue  # Überspringe ungültige Werte

        partial, partial_amplitude = estimate_frequency_and_amplitude(
            np.angle(prev_NOTESIG[k_test]), np.angle(NOTESIG[k_test]), k_test, fft_size, H, sr, frame, window)

        partial_AMP = 20 * np.log10(np.abs(partial_amplitude))

        # Filter nach Frequenzbereich und Mindestamplitude
        if partial_AMP > best_amplitude:
            best_partial = partial
            best_amplitude = partial_AMP

    fundamental = best_partial
    return fundamental


def estBeta(sig, sr, dbg, beta, fundamental):
    stringBetas = []
    beta_max = 10*beta


    W = 4096
    (H, fft_size) = W // 8, W


    buffered_signal = np.lib.stride_tricks.sliding_window_view(sig, window_shape=W)[::H]
    padded_signal = np.pad(buffered_signal, ((0, 0), (0, fft_size - W)), mode='constant')
    window = scipy.signal.windows.hann(fft_size, sym=False)
    window_sum = np.sum(window)  # Sum of the window coefficients
    buffered_windowed_signal = padded_signal * window
    freqs = rfftfreq(fft_size, d=1 / sr)

    W = fft_size



    prev_NOTESIG = rfft(buffered_windowed_signal[0])
    noteBetas = []

    # Lists to store frequencies for plotting
    frame_times = []
    model_freqs = []
    contour_fundamentals = []
    phase_fundamentals = []

    # Normalize
    buffered_windowed_signal /= np.max(np.abs(buffered_windowed_signal))

    for frame_idx, frame in enumerate(buffered_windowed_signal[1:], start=1):
        NOTESIG = rfft(frame)
        magnitude_spectrum = np.abs(NOTESIG)

        # Normalize the magnitude spectrum by 2/window_sum
        magnitude_spectrum_normalized = (2 * magnitude_spectrum) / window_sum

        # Convert to dB
        magnitude_spectrum = 20 * np.log10(magnitude_spectrum_normalized + 1e-10)

        Threshold = -50  # -40 dB
        peak_indices, _ = find_peaks(magnitude_spectrum, height=Threshold)
        peak_freqs = freqs[peak_indices]

        if len(peak_indices) == 0:
            print("no peaks detected")
            continue






        phase_fundamentals.append(fundamental)



        # Calculate partials
        n_partials = min(len(peak_indices), 20)
        partials = calculate_partials(fundamental, int(fundamental * W / sr), prev_NOTESIG, NOTESIG, fft_size, H,
                                      sr, beta_max, n_partials, dbg, frame, window, peak_indices)

        # Calculate Beta for each partial
        partialBetas = [
            ((freq / ((order + 1) * partials[0])) ** 2 - 1) / ((order + 1) ** 2)
            for order, freq in partials.items() if order > 0
        ]

        # Filter outliers
        valid_partialBetas = [pb for pb in partialBetas if 0 < pb < beta_max]
        if valid_partialBetas:
            l = np.array(valid_partialBetas)
            Q1 = np.quantile(l, 0.2)
            Q3 = np.quantile(l, 0.8)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_partialBetas = l[(l >= lower_bound) & (l <= upper_bound)].tolist()

            if cleaned_partialBetas:
                noteBetas.append(cleaned_partialBetas)
            else:
                print("No values left after IQR-filter")
        else:
            print("No valid partial Betas")
            noteBetas.append(valid_partialBetas)

        # Phase Estimaton Logic
        prev_NOTESIG = NOTESIG

        # plot only if toggle
        if dbg:
            plt.figure(figsize=(12, 6))
            # Frequenzspektrum
            plt.plot(freqs, magnitude_spectrum, label=f"Frame {frame_idx}")
            # Threshold-Linie
            plt.axhline(Threshold, color='black', linestyle='--', label=f'Threshold: {Threshold:.2f} dB')
            # Modellierte Grundfrequenz (noteFreq)
            plt.scatter(fundamental, 0, color="purple", marker="x", s=100, label="Model NoteFreq")

            # # Gefundene Peaks
            plt.scatter(freqs[peak_indices], magnitude_spectrum[peak_indices], color='orange', marker="o",
                        label="Detected Peaks")

            # Vorhergesagte Partialfrequenzen (basierend auf fundamental)
            predicted_partials = [fundamental * (i + 1) for i in range(1, n_partials)]
            plt.scatter(predicted_partials, [0] * len(predicted_partials),
                        color='red', marker="s", label="Predicted Partials")
            # Tatsächlich gefundene Partialfrequenzen aus dem Dict partials
            real_partials = list(partials.values())
            plt.scatter(real_partials, [-2] * len(real_partials),
                        color='green', marker="d", label="Real Partials")
            # Achsen und Darstellung
            plt.title(f"Frequency Spectrum (Frame {frame_idx})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.xscale("log")
            plt.grid()
            plt.legend()
            plt.show()

        if noteBetas:
            stringBetas.append(noteBetas)

    return stringBetas


def flatten_recursive(nested_list):
    """ Rekursive Funktion, um eine verschachtelte Liste oder ein Array zu flachen. """
    flat_list = []
    if isinstance(nested_list, (list, np.ndarray)):  # Überprüfen, ob es ein iterierbares Objekt ist
        for item in nested_list:
            # Wenn das Item wiederum iterierbar ist, dann rekursiv flach machen
            if isinstance(item, (list, np.ndarray)):
                flat_list.extend(flatten_recursive(item))  # Rekursiv flach machen
            else:
                # Wenn es kein iterierbares Objekt ist, einfach den Wert hinzufügen
                flat_list.append(item)
    else:
        # Falls das übergebene Objekt keine Liste oder Array ist, direkt hinzufügen
        flat_list.append(nested_list)

    return flat_list


def noteToFreq(note):
    a = 440  # frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def get_annotation_filename(audio_filename):
    # Extrahiere die Base, behalte `solo` oder `comp`, entferne `pshift`
    base_name = audio_filename.split(".")[0]

    if "_pshift" in base_name:
        base_name = base_name.split("_pshift")[0]

    # Entferne die Audio-spezifischen Endungen (wie "_cln" oder "_mic")
    for suffix in ["_hex", "_cln", "_mic", "_debleeded"]:
        base_name = base_name.replace(suffix, "")

    # Füge das notwendige Format für die Annotation an
    return f"{base_name}_notes.npy"


def generate_signal(fundamental, n_partials, sr, beta, length):
    t = np.arange(length) / sr  # Zeitachse
    sig = np.zeros(length)  # Signal initialisieren

    # Fundamental (erste Frequenz bleibt unverändert)
    sig += np.sin(2 * np.pi * fundamental * t)

    # Obertöne (ab der zweiten Frequenz)
    for i in range(2, n_partials + 1):  # Beginnt bei i=2 für Obertöne
        # Berechne die Frequenz des aktuellen Obertone (inkl. Inharmonizität)
        partial_freq = i * fundamental * math.sqrt(1 + beta * (i) ** 2)

        # Berechne den Amplitudenabfall (typischerweise fallend)
        amplitude = 1 / (i ** 0.5)  # Beispielhafte Amplitudenmodifikation (kann angepasst werden)

        # Generiere das Sinussignal für diesen Partialton
        sig += amplitude * np.sin(2 * np.pi * partial_freq * t)

    return sig

# Main
def main():

    # generate Audio-File
    length = 88200  # (2 Sekunden)
    beta = 1.4e-4
    n_partials = 25
    sr = 44100
    fundamental = 440

    sig = generate_signal(fundamental, n_partials, sr, beta, length)
    sig /= np.max(abs(sig))

    dbg = 0

    estimated_betas = estBeta(sig, sr, dbg, beta, fundamental)
    betas = flatten_recursive(estimated_betas)

    # Statistische Auswertung
    mean_beta = np.mean(betas)
    std_beta = np.std(betas, ddof=1)   # Stichproben-Standardabweichung
    var_beta = np.var(betas, ddof=1)   # Stichprobenvarianz

    print("=== Beta-Statistik ===")
    print(f"Anzahl Werte: {len(betas)}")
    print(f"Estimated Beta (Mittelwert): {mean_beta:.6e}")
    print(f"Standardabweichung: {std_beta:.6e}")
    print(f"Varianz: {var_beta:.6e}")

    # Histogram plotten
    plt.figure(figsize=(8, 5))
    plt.hist(betas, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    plt.axvline(mean_beta, color="red", linestyle="--", label=f"Mittelwert = {mean_beta:.2e}")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Häufigkeit")
    plt.title("Verteilung der geschätzten Beta-Werte")
    plt.legend()
    plt.xlim(0, 0.0001)
    plt.grid(True, alpha=0.3)
    plt.show()



# %%
if __name__ == "__main__":
    main()