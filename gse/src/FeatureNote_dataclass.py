from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Features:
    inharmonicity: Optional[float] = None
    spectral_flux: Optional[float] = None
    spectral_centroid: Optional[float] = None

@dataclass
class GT:
    pitch: Optional[float] = None # midi pitch
    string: Optional[int] = None
    is_drum: Optional[bool] = None
    program: Optional[int] = None
    onset: Optional[float] = None
    offset: Optional[float] = None
    midi_note: Optional[int] = None
    velocity: Optional[int] = None
    contour: List[Tuple[float, float]] = field(default_factory=list)
    string_index: Optional[int] = None

    # def __post_init__(self):


@dataclass
class FeatureNote:
    # Core audio noteData
    audio: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    source_file: Optional[str] = None
    source_path: Optional[str] = None

    # Matching / bookkeeping
    match_note: Optional[bool] = None

    # Ground truth & computed noteData
    gt: Optional[GT] = None
    features: Optional[Features] = None

    # Estimations
    estimated_string: Optional[int] = None