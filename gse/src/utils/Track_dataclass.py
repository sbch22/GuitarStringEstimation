from dataclasses import dataclass, field
from typing import List, Optional
from gse.src.utils.FeatureNote_dataclass import FeatureNote
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
                key=lambda note: (note.attributes.onset, note.attributes.is_drum, note.attributes.program, note.attributes.velocity, note.attributes.pitch, note.attributes.offset))
        return notes

    @staticmethod
    def validate_notes(notes: List["FeatureNote"], fix: bool = True) -> List["FeatureNote"]:
        """Validate and optionally fix unrealistic notes (using note.attributes attributes)."""
        if not notes:
            return notes

        # Create a new list so we can safely remove elements while iterating
        valid_notes = []

        for note in notes:
            attributes = note.attributes  # shortcut

            # Skip notes with no ground-truth object
            if attributes is None:
                continue

            # If onset is missing, skip or drop
            if attributes.onset is None:
                if fix:
                    warnings.warn(f"Dropping note without onset: {note}")
                continue

            # If offset is missing, create a minimal one
            if attributes.offset is None:
                if fix:
                    attributes.offset = attributes.onset + MINIMUM_OFFSET_TIME

            # If onset > offset, fix
            elif attributes.onset > attributes.offset:
                warnings.warn(f"Note onset > offset: {note}")
                if fix:
                    attributes.offset = attributes.onset + MINIMUM_OFFSET_TIME
                    print(f"Fixed! Setting offset = onset + {MINIMUM_OFFSET_TIME:.3f}s")

            # If too short (non-drum), fix
            elif attributes.is_drum is False and attributes.offset - attributes.onset < MINIMUM_OFFSET_TIME:
                if fix:
                    attributes.offset = attributes.onset + MINIMUM_OFFSET_TIME

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
            (n.attributes.pitch, n.attributes.program, n.attributes.is_drum)
            for n in notes
            if n.attributes is not None
        )

        for pitch, program, is_drum in channels:
            # Notes in this channel
            channel_notes = [
                n for n in notes
                if n.attributes and n.attributes.pitch == pitch and n.attributes.program == program and n.attributes.is_drum == is_drum
            ]
            sorted_notes = sorted(channel_notes, key=lambda n: n.attributes.onset)

            # Trim overlapping offsets
            for i in range(1, len(sorted_notes)):
                if sorted_notes[i - 1].attributes.offset > sorted_notes[i].attributes.onset:
                    sorted_notes[i - 1].attributes.offset = sorted_notes[i].attributes.onset

            # Keep valid (nonzero) notes
            valid_notes = [
                n for n in sorted_notes
                if n.attributes.onset < n.attributes.offset
            ]

            trimmed_notes.extend(valid_notes)

        if sort:
            trimmed_notes.sort(
                key=lambda n: (
                    n.attributes.onset,
                    n.attributes.is_drum,
                    n.attributes.program,
                    n.attributes.pitch
                )
            )

        return trimmed_notes

    @staticmethod
    def match_notes(delta: float, all_notes: List['FeatureNote']):
        """
        Match predicted notes with ground-truth notes based on onset tolerance and pitch & program equality.
        Marks both notes as matched by setting `Track.match = True`.
        """
        # Separate GT and predicted notes
        gt_notes = [n for n in all_notes if n.origin == "gt"]
        pred_notes = [n for n in all_notes if n.origin == "model"]

        for pred in pred_notes:
            for gt in gt_notes:
                if gt.attributes.midi_note != pred.attributes.midi_note:
                    continue
                if gt.attributes.program != pred.attributes.program:
                    continue
                time_diff = abs(pred.attributes.onset - gt.attributes.onset)
                if time_diff <= delta:
                    best_pred = pred
                    pred.match = True
                    gt.match = True
