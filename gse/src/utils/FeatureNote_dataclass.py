from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math

@dataclass
class Features:
    betas: List[float] = field(default_factory=list)
    inharmonicity: Optional[float] = None
    spectral_flux: Optional[float] = None
    spectral_centroid: Optional[float] = None

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


@dataclass
class FeatureNote:
    # Matching / bookkeeping
    origin: Optional[str] = None # GT, model or match
    valid: bool = False
    filter_reason: Optional[str] = None # list of strings

    # Attributes: pitch, onset, offset, etc.
    attributes: Optional[Attributes] = None
    # audio feature vector
    features: Optional[Features] = None

    # Estimations
    estimated_string: Optional[int] = None

    partials: Optional[Partials] = None

    def delete_from(self, notes: list):
        """
        Remove this note from a list of FeatureNote objects.
        """
        if self in notes:
            notes.remove(self)

    def what_fret(self):
        """
        Calculate Fret from String and f0
        """
        # source: https://de.wikipedia.org/wiki/Stimmen_einer_Gitarre
        strings_f0 = {
            0: 82.42,  # E2
            1: 110.0,  # A2
            2: 146.83,  # D3
            3: 196.00,  # G3
            4: 246.94,  # B3
            5: 329.63  # E4
        }
        string_f0 = strings_f0[self.attributes.string_index]
        f0 = self.attributes.pitch

        fret = 12 * math.log2(f0 / string_f0)

        # auf nächsten Bund runden
        fret = int(round(fret))

        # optional clampen: 0 < fret < 24
        if not (0 <= fret <= 24):
            # print("Unplayable:", self.attributes.pitch, "string", self.attributes.string_index)
            self.attributes.fret = None
            self.valid = False
            self.filter_reason = 'fretting invalid'
            return

        self.attributes.fret = fret
