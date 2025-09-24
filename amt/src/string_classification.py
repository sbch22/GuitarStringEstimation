import os
import sys

from mpmath import linspace

sys.path.append(os.path.abspath(''))

from collections import Counter
import argparse
import torch

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
from typing import Tuple, Dict, Literal, List, Optional
import torchaudio
import sounddevice as sd
from scipy.stats import norm
import dataclasses
from utils.note_event_dataclasses import matchNote, stringNote
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import List
from scipy.stats import gaussian_kde, norm, wasserstein_distance


# import from betaDistributions
from betaDistributions import (
    noteToFreq,

)


""" Function with plotting, just wasserstein"""
def wasserstein(stringNotes, betas_roh, use_kde=True, plot=False):
    """
    Predicts the most likely string for each note using Wasserstein distance
    between note KDE and normal distributions.

    Args:
        stringNotes (list): List of note objects containing noteBetas.
        betas_roh (dict): Dictionary of raw beta values per string.
        use_kde (bool): Whether to use KDE for noteBetas.
        plot (bool): Whether to plot distributions for debugging.

    Returns:
        list: Updated list of note objects with string predictions.
    """
    updated_stringNotes = []
    string_keys = [f"Saite_{i + 1}" for i in range(6)]
    string_labels = [f"String_{i}" for i in range(6)]  # Anpassung der Labels für Plots

    # Berechne Mittelwert und Standardabweichung für jede Saite
    betas_mean_std = {
        key: (np.mean(vals), max(1e-6, np.std(vals))) if len(vals) > 1 else (vals[0], 1e-6)
        for key, vals in betas_roh.items()
    }

    # Farben für die Saiten
    colors = {
        "Saite_1": "skyblue",
        "Saite_2": "red",
        "Saite_3": "green",
        "Saite_4": "orange",
        "Saite_5": "purple",
        "Saite_6": "brown",
    }


    # Plot setup for string distributions
    if plot:
        plt.figure(figsize=(12, 8))
        x_vals = np.linspace(0, 0.001, 200)

        # Plot distributions for each string
        for i, string_key in enumerate(string_keys):
            mean, std = betas_mean_std[string_key]
            string_pdf = norm.pdf(x_vals, mean, std)
            plt.plot(x_vals, string_pdf, label=f"{string_labels[i]} (Normal)", color=colors[string_key])

        plt.xlabel("Beta Value")
        plt.ylabel("Probability Density")
        plt.title("String PDFs")
        plt.legend()
        plt.grid()
        plt.show()

    for note in stringNotes:
        best_string = None
        likelihood_ratio = None
        wasserstein_scores = []
        freq_weights = []

        flat_noteBetas = note.noteBetas
        noteFreq = noteToFreq(note.pitch)

        if flat_noteBetas is not None and len(flat_noteBetas) > 1:
            # KDE für noteBetas berechnen
            x_vals = np.linspace(0, 0.001, 200)
            note_kde = gaussian_kde(flat_noteBetas, bw_method="silverman")
            note_kde_values = note_kde(x_vals)

            for string_idx, string_key in enumerate(string_keys):
                mean, std = betas_mean_std[string_key]

                # Erstelle Normalverteilungs-PDF für die Saite
                string_pdf = norm.pdf(x_vals, mean, std)

                # Berechne Wasserstein-Distanz zwischen KDE und der Normalverteilung
                distance = wasserstein_distance(note_kde_values, string_pdf)

                wasserstein_scores.append(distance)



            if wasserstein_scores:

                best_string_idx = np.argmin(wasserstein_scores)  # Kleinste Distanz ist beste Übereinstimmung
                best_string = best_string_idx

                if len(wasserstein_scores) > 1:
                    sorted_scores = np.sort(wasserstein_scores)
                    likelihood_ratio = sorted_scores[1] - sorted_scores[0]  # Differenz der besten zwei

                if plot:
                    plt.figure(figsize=(8, 5))
                    matched_string_key = string_keys[best_string_idx]
                    mean, std = betas_mean_std[matched_string_key]
                    matched_string_pdf = norm.pdf(x_vals, mean, std)

                    # Plot noteBeta KDE
                    plt.plot(x_vals, note_kde_values, label=f"Note {note.pitch} KDE", color="black", linewidth=2)

                    # Plot matched string distribution mit aktualisiertem Label
                    plt.plot(
                        x_vals,
                        matched_string_pdf,
                        label=f"{string_labels[best_string_idx]} (Normal)",
                        color=colors[matched_string_key],
                        linestyle="--"
                    )

                    # Highlight the maximum of the noteBeta KDE
                    max_x = x_vals[np.argmax(note_kde_values)]  # x-Wert des Maximums
                    max_y = np.max(note_kde_values)  # y-Wert des Maximums
                    plt.scatter(max_x, max_y, color="red", zorder=5, label="KDE Max")

                    plt.xlabel("Beta Value")
                    plt.ylabel("Probability Density")
                    plt.title(f"Note {note.pitch} KDE vs Matched String ({string_labels[best_string_idx]})")
                    plt.legend()
                    plt.grid()
                    plt.show()

        updated_note = stringNote(
            is_drum=note.is_drum,
            program=note.program,
            onset=note.onset,
            offset=note.offset,
            pitch=note.pitch,
            velocity=note.velocity,
            noteBetas=note.noteBetas,
            string_pred=best_string,  # Beste Saite basierend auf Wasserstein-Distanz
            likelihood_ratio=likelihood_ratio,  # Sicherheit der Vorhersage
            stringGT=note.stringGT
        )

        updated_stringNotes.append(updated_note)

    return updated_stringNotes

"""theoretische Modell"""
def freq_theoretical(noteFreq, string_freq):
    if noteFreq <= (string_freq / math.pow(2, 1 / 24)):
        return 0
    elif noteFreq <= (4/3) * string_freq:
        return 1.0
    elif noteFreq <= (34/9) * string_freq:
        # Konstanter linearer Abfall ab der Quarte bis 34/9 * string_freq
        max_drop = 1  # Maximaler Abfall von 1.0 auf 0.0
        slope = -max_drop / (((34/9) * string_freq) - ((4/3) * string_freq))
        return 1.0 + slope * (noteFreq - (4/3) * string_freq)
    else:
        # Konstanter niedriger Wert über 34/9 * string_freq
        return 0

"""Custom: semi-theoretisch, bisher bestes Ergebnis"""
def freq_semi_empirical(noteFreq, string_freq):
    if noteFreq <= (string_freq / math.pow(2, 1 / 24)):  # If note frequency is below or equal to the string frequency
        return 0  # Very likely
    elif noteFreq <= (5/3) * string_freq:  # If frequency is within the fifth (3:2 ratio)
        return 1.0  # Still very likely
    elif noteFreq <= 2 * string_freq:  # If frequency is between the fifth and double the string frequency
        # Exponential decay function for frequencies between the fifth and double the string frequency
        return np.exp(-((noteFreq - (5/3) * string_freq) ** 2) / (2 * (0.2 * string_freq) ** 2))
    else:  # Frequency above double the string frequency
        return 1e-6  # Very unlikely

"""Funktion combined, Wasserstein - best results"""
def wasserstein_freq_semi_empirical(stringNotes, betas_roh, use_kde=True, plot=False):
    updated_stringNotes = []
    string_keys = [f"Saite_{i + 1}" for i in range(6)]
    string_labels = [f"String_{i}" for i in range(6)]  # Anpassung der Labels für Plots

    betas_mean_std = {
        key: (np.mean(vals), max(1e-6, np.std(vals))) if len(vals) > 1 else (vals[0], 1e-6)
        for key, vals in betas_roh.items()
    }

    string_frequencies = {
        "Saite_1": 82.41,  # E2
        "Saite_2": 110.00,  # A2
        "Saite_3": 146.83,  # D3
        "Saite_4": 196.00,  # G3
        "Saite_5": 246.94,  # B3
        "Saite_6": 329.63,  # E4
    }

    #Farben für die Saiten
    colors = {
        "Saite_1": "skyblue",
        "Saite_2": "red",
        "Saite_3": "green",
        "Saite_4": "orange",
        "Saite_5": "purple",
        "Saite_6": "brown",
    }


    # Plot setup for string distributions
    if plot:
        plt.figure(figsize=(12, 8))
        x_vals = np.linspace(0, 0.001, 200)

        # Plot distributions for each string
        for i, string_key in enumerate(string_keys):
            mean, std = betas_mean_std[string_key]
            string_pdf = norm.pdf(x_vals, mean, std)
            plt.plot(x_vals, string_pdf, label=f"{string_labels[i]} (Normal)", color=colors[string_key])

        plt.xlabel("Beta Value")
        plt.ylabel("Probability Density")
        plt.title("String PDFs")
        plt.legend()
        plt.grid()
        plt.show()

    for note in stringNotes:
        best_string = None
        wasserstein_scores = []
        freq_weights = []

        flat_noteBetas = note.noteBetas
        noteFreq = noteToFreq(note.pitch)

        if flat_noteBetas is not None and len(flat_noteBetas) > 1:
            x_vals = np.linspace(0, 0.001, 1000)
            note_kde = gaussian_kde(flat_noteBetas, bw_method="silverman")
            note_kde_values = note_kde(x_vals)
            # note_kde_values /= max(note_kde_values)

            for string_idx, string_key in enumerate(string_keys):
                mean, std = betas_mean_std[string_key]
                string_pdf = norm.pdf(x_vals, mean, std)



                distance = wasserstein_distance(note_kde_values, string_pdf)
                wasserstein_scores.append(distance)

                string_freq = string_frequencies[string_key]
                freq_weight = freq_semi_empirical(noteFreq, string_freq)
                freq_weights.append(freq_weight)

            wasserstein_scores = np.array(wasserstein_scores)
            freq_weights = np.array(freq_weights)

            if wasserstein_scores.max() > wasserstein_scores.min():
                wasserstein_scores_norm = (wasserstein_scores - wasserstein_scores.min()) / (
                        wasserstein_scores.max() - wasserstein_scores.min())
            else:
                wasserstein_scores_norm = np.zeros_like(wasserstein_scores)

            if freq_weights.max() > freq_weights.min():
                freq_weights_norm = (freq_weights - freq_weights.min()) / (
                        freq_weights.max() - freq_weights.min())
            else:
                freq_weights_norm = np.zeros_like(freq_weights)

            wasserstein_weight = 1
            freq_weight_weight = 1 #* 5 # - wasserstein_weight

            combined_scores = wasserstein_weight * (-wasserstein_scores_norm) + freq_weight_weight * freq_weights_norm
            best_string_idx = np.argmax(combined_scores)
            best_string = best_string_idx

            if plot:
                plt.figure(figsize=(8, 5))
                matched_string_key = string_keys[best_string_idx]
                mean, std = betas_mean_std[matched_string_key]
                matched_string_pdf = norm.pdf(x_vals, mean, std)

                # Plot noteBeta KDE
                plt.plot(x_vals, note_kde_values, label=f"Note {note.pitch} KDE", color="black", linewidth=2)

                # Plot matched string distribution mit aktualisiertem Label
                plt.plot(
                    x_vals,
                    matched_string_pdf,
                    label=f"{string_labels[best_string_idx]} (Normal)",
                    color=colors[matched_string_key],
                    linestyle="--"
                )

                # Highlight the maximum of the noteBeta KDE
                max_x = x_vals[np.argmax(note_kde_values)]  # x-Wert des Maximums
                max_y = np.max(note_kde_values)  # y-Wert des Maximums
                plt.scatter(max_x, max_y, color="red", zorder=5, label="KDE Max")

                plt.xlabel("Beta Value")
                plt.ylabel("Probability Density")
                plt.title(f"Note {note.pitch} KDE vs Matched String ({string_labels[best_string_idx]})")
                plt.legend()
                plt.grid()
                plt.show()

        updated_note = stringNote(
            is_drum=note.is_drum,
            program=note.program,
            onset=note.onset,
            offset=note.offset,
            pitch=note.pitch,
            velocity=note.velocity,
            noteBetas=note.noteBetas,
            string_pred=best_string,
            likelihood_ratio=None,
            stringGT=note.stringGT
        )
        updated_stringNotes.append(updated_note)

    return updated_stringNotes