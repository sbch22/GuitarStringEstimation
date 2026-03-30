from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import math
from enum import Enum


@dataclass
class Features:
    beta: Optional[float] = None
    f0: Optional[float] = None
    betas_measures: Optional[np.ndarray] = None
    valid_partials: Optional[np.ndarray] = None
    rel_partial_amplitudes: Optional[np.ndarray] = None
    amp_decay_coefficients: Optional[np.ndarray] = None
    rel_freq_deviations: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None

    feature_vector: Optional[np.ndarray] = None

    def fill_feature_vector(self) -> None:
        segments = {
            "f0":                     self.f0,
            "betas_measures":         self.betas_measures,
            "valid_partials":         self.valid_partials,
            "spectral_centroid":      self.spectral_centroid,
            "rel_partial_amplitudes": self.rel_partial_amplitudes,
            "amp_decay_coefficients": self.amp_decay_coefficients,
            "rel_freq_deviations":    self.rel_freq_deviations,
        }

        missing = [k for k, v in segments.items() if v is None]
        if missing:
            raise ValueError(f"Features not yet computed: {missing}")

        self.feature_vector = np.concatenate(
            [np.asarray(v, dtype=float).ravel() for v in segments.values()],
            axis=0
        )

    def segment_layout(self) -> dict[str, int]:
        """Returns {segment_name: length} in the same order as fill_feature_vector."""
        segments = {
            "f0": self.f0,
            "betas_measures": self.betas_measures,
            "valid_partials": self.valid_partials,
            "spectral_centroid": self.spectral_centroid,
            "rel_partial_amplitudes": self.rel_partial_amplitudes,
            "amp_decay_coefficients": self.amp_decay_coefficients,
            "rel_freq_deviations": self.rel_freq_deviations,
        }
        return {k: np.asarray(v, dtype=float).ravel().size for k, v in segments.items()}

@dataclass
class Attributes:
    pitch: Optional[float] = None # frequency in Hz
    is_drum: Optional[bool] = None
    program: Optional[int] = None
    onset: Optional[float] = None
    offset: Optional[float] = None
    midi_note: Optional[int] = None
    velocity: Optional[int] = None
    contour: List[Tuple[float, float]] = field(default_factory=list)
    string_index: Optional[int] = None
    fret: Optional[float] = None

@dataclass
class Partials:
    frametimes: np.ndarray[float]
    frequencies: np.ndarray[float]
    amplitudes: np.ndarray[float]

class FilterReason(Enum):
    # preprocessing
    FRETTING_INVALID = "what_fret found invalid fret"
    NO_STRING = "no string found by model"
    MISMATCH_BETWEEN_STRINGS = "note assignment likely wring between strings due to bleed"
    NO_MATCH = "no matching note could be found between model output and ground truth"
    # find_partials.py
    NO_NOTE_AUDIO = "no note audio found from onsets"
    NO_HARMONIC_AUDIO = "no harmonic audio after extraction"
    HARMONIC_AUDIO_TOO_SHORT = "harmonic audio too short"
    NO_PARTIALS_FOUND = "no partials found"
    NO_NOTE = "houston we have a problem"

    # calculate_features.py
    FEATURE_EXTRACTION_FAILED = "feature extraction failed"
    NO_BETAS = "no betas from calculations"
    NO_BETAS_AFTER_FILTER = "no betas after filtering"
    NO_FEATURES = "no features"


@dataclass
class FeatureNote:
    origin: Optional[str] = None # GT, model or match
    dataset: Optional[str] = None
    valid: bool = False
    match: bool = False
    filter_reason: Optional[FilterReason] = None # list of strings
    filter_step: Optional[str] = None

    attributes: Optional[Attributes] = None
    features: Dict[str, Features] = field(default_factory=dict)
    partials: Dict[str, Partials] = field(default_factory=dict)  # keyed by audio_type
    string_probs: Optional[np.ndarray] = None

    def invalidate(self, reason: FilterReason, step: str) -> None:
        """
        Mark this note as invalid.
        Because Track.valid_notes is a @property derived from Track.notes,
        this single call is all you need — no manual list surgery required.
        """
        self.valid = False
        self.filter_reason = reason
        self.filter_step = step

    def delete_from(self, notes: list):
        """
        Remove this note from a list of FeatureNote objects.
        """
        if self in notes:
            notes.remove(self)

    def what_fret(self) -> None:
        """Calculate and assign the fret number from string_index and pitch (Hz)."""
        STRING_OPEN_FREQS = {
            0: 82.42,   # E2
            1: 110.00,  # A2
            2: 146.83,  # D3
            3: 196.00,  # G3
            4: 246.94,  # B3
            5: 329.63,  # E4
        }

        if self.attributes.string_index is None:
            self.invalidate(FilterReason.NO_STRING, step="what_fret")
            return

        string_f0 = STRING_OPEN_FREQS[self.attributes.string_index]
        f0 = self.attributes.pitch
        fret = int(round(12 * math.log2(f0 / string_f0)))

        if not (0 <= fret <= 24):
            self.attributes.fret = None
            self.invalidate(FilterReason.FRETTING_INVALID, step="what_fret")
            return

        self.attributes.fret = fret
