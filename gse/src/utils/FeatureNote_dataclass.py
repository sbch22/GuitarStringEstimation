from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Features:
    betas: List[float] = field(default_factory=list)
    inharmonicity: Optional[float] = None
    spectral_flux: Optional[float] = None
    spectral_centroid: Optional[float] = None

@dataclass
class Attributes:
    pitch: Optional[float] = None # frequency in Hz
    string: Optional[int] = None
    is_drum: Optional[bool] = None
    program: Optional[int] = None
    onset: Optional[float] = None
    offset: Optional[float] = None
    midi_note: Optional[int] = None
    velocity: Optional[int] = None
    contour: List[Tuple[float, float]] = field(default_factory=list)
    string_index: Optional[int] = None

    @staticmethod
    def fill_missing_attributes(features: List[Features]) -> List[Features]:
        # Todo: fill attributes like midi -> pitch, string name etc.
        return features



@dataclass
class Partial:
    n: int
    frametimes: np.ndarray[float]
    frequencies: np.ndarray[float]
    amplitudes: np.ndarray[float]


@dataclass
class FeatureNote:
    # Matching / bookkeeping
    origin: Optional[str] = None # GT, model or match
    match: bool = False

    # Attributes: pitch, onset, offset, etc.
    attributes: Optional[Attributes] = None

    # audio feature vector
    features: Optional[Features] = None

    # Estimations
    estimated_string: Optional[int] = None

    partials: List["Partial"] = field(default_factory=list)

    # TODO: method that returns windowed frames of pure data -> vectorized operations