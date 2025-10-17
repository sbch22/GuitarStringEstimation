from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from FeatureNote_dataclass import FeatureNote
import pickle
import warnings
import pyfar as pf

# taken from YMT3
DRUM_OFFSET_TIME = 0.01  # in seconds
MINIMUM_OFFSET_TIME = 0.01  # this is used to avoid zero-length notes
DRUM_PROGRAM = 128

@dataclass
class TrackAudio:
    mono_mic: Optional[pf.Signal] = None
    hex_debleeded: Optional[pf.Signal] = None
    hex_mono_mix: Optional[pf.Signal] = None


@dataclass
class Track:
    name: str
    audio: Optional[TrackAudio] = None
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

    @staticmethod
    def sort_notes(notes: List[FeatureNote]) -> List[FeatureNote]:
        """ Sort notes by increasing order of onsets, and at the same timing, by increasing order of program and pitch. """
        if len(notes) > 0:
            notes.sort(
                key=lambda note: (note.gt.onset, note.gt.is_drum, note.gt.program, note.gt.velocity, note.gt.pitch, note.gt.offset))
        return notes

    @staticmethod
    def validate_notes(notes: List["FeatureNote"], fix: bool = True) -> List["FeatureNote"]:
        """Validate and optionally fix unrealistic notes (using note.gt attributes)."""
        if not notes:
            return notes

        # Create a new list so we can safely remove elements while iterating
        valid_notes = []

        for note in notes:
            gt = note.gt  # shortcut

            # Skip notes with no ground-truth object
            if gt is None:
                continue

            # If onset is missing, skip or drop
            if gt.onset is None:
                if fix:
                    warnings.warn(f"Dropping note without onset: {note}")
                continue

            # If offset is missing, create a minimal one
            if gt.offset is None:
                if fix:
                    gt.offset = gt.onset + MINIMUM_OFFSET_TIME

            # If onset > offset, fix
            elif gt.onset > gt.offset:
                warnings.warn(f"Note onset > offset: {note}")
                if fix:
                    gt.offset = gt.onset + MINIMUM_OFFSET_TIME
                    print(f"Fixed! Setting offset = onset + {MINIMUM_OFFSET_TIME:.3f}s")

            # If too short (non-drum), fix
            elif gt.is_drum is False and gt.offset - gt.onset < MINIMUM_OFFSET_TIME:
                if fix:
                    gt.offset = gt.onset + MINIMUM_OFFSET_TIME

            valid_notes.append(note)

        return valid_notes

    @staticmethod
    def trim_overlapping_notes(notes: List["FeatureNote"], sort: bool = True) -> List["FeatureNote"]:
        """Trim overlapping notes and drop zero-length ones, using note.gt attributes."""
        if len(notes) <= 1:
            return notes

        trimmed_notes = []

        # Identify unique "channels" (pitch, program, drum)
        channels = set(
            (n.gt.pitch, n.gt.program, n.gt.is_drum)
            for n in notes
            if n.gt is not None
        )

        for pitch, program, is_drum in channels:
            # Notes in this channel
            channel_notes = [
                n for n in notes
                if n.gt and n.gt.pitch == pitch and n.gt.program == program and n.gt.is_drum == is_drum
            ]
            sorted_notes = sorted(channel_notes, key=lambda n: n.gt.onset)

            # Trim overlapping offsets
            for i in range(1, len(sorted_notes)):
                if sorted_notes[i - 1].gt.offset > sorted_notes[i].gt.onset:
                    sorted_notes[i - 1].gt.offset = sorted_notes[i].gt.onset

            # Keep valid (nonzero) notes
            valid_notes = [
                n for n in sorted_notes
                if n.gt.onset < n.gt.offset
            ]

            trimmed_notes.extend(valid_notes)

        if sort:
            trimmed_notes.sort(
                key=lambda n: (
                    n.gt.onset,
                    n.gt.is_drum,
                    n.gt.program,
                    n.gt.pitch
                )
            )

        return trimmed_notes