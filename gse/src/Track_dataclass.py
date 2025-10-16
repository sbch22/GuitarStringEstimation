from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from FeatureNote_dataclass import FeatureNote
import pickle

@dataclass
class Track:
    name: str
    audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    notes: List["FeatureNote"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save the track (and its notes) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Track":
        """Load a saved track."""
        with open(path, "rb") as f:
            return pickle.load(f)