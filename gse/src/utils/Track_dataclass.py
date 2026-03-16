from dataclasses import dataclass, field
from typing import List, Optional
from gse.src.utils.FeatureNote_dataclass import FeatureNote, FilterReason
import pickle
import warnings
import pyfar as pf
import numpy as np
from collections import defaultdict

# -------- constants ---------  taken from YMT3
DRUM_OFFSET_TIME = 0.01  # in seconds
MINIMUM_OFFSET_TIME = 0.01  # this is used to avoid zero-length notes
DRUM_PROGRAM = 128

# ── standalone utility ────────────────────────────────────────────────────────
def filter_analysis(notes: List[FeatureNote], step: str = None) -> dict:
    """
    Print a breakdown of valid / invalid notes and return the breakdown dict.

    Call this after *every* pipeline step to see exactly how many notes were
    filtered, why, and at which stage.

    Parameters
    ----------
    notes : list of FeatureNote
        Pass ``track.notes`` — the full archive including invalids.
    step  : str, optional
        Label printed in the header (e.g. ``"match_notes"``).

    Returns
    -------
    dict  {(filter_step, reason_value): count}
    """
    total   = len(notes)
    valid   = [n for n in notes if n.valid]
    invalid = [n for n in notes if not n.valid]

    breakdown: dict = defaultdict(int)
    for note in invalid:
        s      = getattr(note, "filter_step",   "unknown_step")
        r      = getattr(note, "filter_reason", None)
        reason = r.value if isinstance(r, FilterReason) else str(r)
        breakdown[(s, reason)] += 1

    header = f" [{step}]" if step else ""
    print(f"\n{'=' * 55}")
    print(f"Filter Analysis{header}")
    print(f"  Total : {total:>5}")
    print(f"  Valid : {len(valid):>5}")
    print(f"  Invalid: {len(invalid):>4}")

    if breakdown:
        print(f"\n  Breakdown by step + reason:")
        for (s, reason), count in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total else 0
            print(f"    [{s}] {reason}: {count}  ({pct:.1f} %)")
    print("=" * 55)

    return dict(breakdown)


@dataclass
class Track:
    name: str
    audio_paths: Optional[dict] = None
    dataset: Optional[str] = None
    notes: List["FeatureNote"] = field(default_factory=list)  # all notes, both origins — untouched archive
    # valid_notes: List["FeatureNote"] = field(default_factory=list)  # model notes surviving all pipeline steps
    gt_notes: List["FeatureNote"] = field(default_factory=list)  # frozen GT f
    metadata: dict = field(default_factory=dict)

    # ── valid_notes: always derived, never stored separately ──────────────────
    @property
    def valid_notes(self) -> List[FeatureNote]:
        """All notes that are currently marked valid."""
        if self.dataset == "single_note":
            return [n for n in self.notes]
        else:
            return [n for n in self.notes if n.valid and n.origin == 'model']


    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Track":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── note helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def sort_notes(notes: List[FeatureNote]) -> List[FeatureNote]:
        """Sort by onset, then program, velocity, pitch, offset."""
        notes.sort(key=lambda n: (
            n.attributes.onset,
            n.attributes.is_drum,
            n.attributes.program,
            n.attributes.velocity,
            n.attributes.pitch,
            n.attributes.offset,
        ))
        return notes

    @staticmethod
    def validate_notes(notes: List[FeatureNote], fix: bool = True) -> List[FeatureNote]:
        """Validate / fix unrealistic note timings in-place; returns a clean list."""
        result = []
        for note in notes:
            attr = note.attributes
            if attr is None or attr.onset is None:
                if fix:
                    warnings.warn(f"Dropping note without onset: {note}")
                continue

            if attr.offset is None:
                if fix:
                    attr.offset = attr.onset + MINIMUM_OFFSET_TIME
            elif attr.onset > attr.offset:
                warnings.warn(f"Note onset > offset: {note}")
                if fix:
                    attr.offset = attr.onset + MINIMUM_OFFSET_TIME
            elif not attr.is_drum and attr.offset - attr.onset < MINIMUM_OFFSET_TIME:
                if fix:
                    attr.offset = attr.onset + MINIMUM_OFFSET_TIME

            result.append(note)
        return result

    @staticmethod
    def trim_overlapping_notes(notes: List[FeatureNote], sort: bool = True) -> List[FeatureNote]:
        """Trim overlapping offsets per (pitch, program, is_drum) channel."""
        if len(notes) <= 1:
            return notes

        channels = {
            (n.attributes.pitch, n.attributes.program, n.attributes.is_drum)
            for n in notes if n.attributes
        }
        trimmed: List[FeatureNote] = []

        for pitch, program, is_drum in channels:
            ch_notes = sorted(
                [n for n in notes
                 if n.attributes
                 and n.attributes.pitch == pitch
                 and n.attributes.program == program
                 and n.attributes.is_drum == is_drum],
                key=lambda n: n.attributes.onset,
            )
            for i in range(1, len(ch_notes)):
                if ch_notes[i - 1].attributes.offset > ch_notes[i].attributes.onset:
                    ch_notes[i - 1].attributes.offset = ch_notes[i].attributes.onset
            trimmed.extend(n for n in ch_notes if n.attributes.onset < n.attributes.offset)

        if sort:
            trimmed.sort(key=lambda n: (
                n.attributes.onset, n.attributes.is_drum,
                n.attributes.program, n.attributes.pitch,
            ))
        return trimmed

    # ── pipeline steps ────────────────────────────────────────────────────────

    @staticmethod
    def match_notes(track: "Track", delta: float) -> None:
        gt_notes = [n for n in track.notes if n.origin == "gt"]
        pred_notes = [n for n in track.notes if n.origin == "model"]

        track.gt_notes = gt_notes  # ALL gt notes — matched ones will have match=True

        for pred in pred_notes:
            matched_gt = next(
                (gt for gt in gt_notes
                 if gt.attributes.midi_note == pred.attributes.midi_note
                 and gt.attributes.program == pred.attributes.program
                 and abs(pred.attributes.onset - gt.attributes.onset) <= delta),
                None,
            )
            if matched_gt:
                pred.valid = True
                pred.match = True
                pred.dataset = matched_gt.dataset
                matched_gt.attributes.string_index = pred.attributes.string_index
                matched_gt.match = True
            else:
                pred.invalidate(FilterReason.NO_MATCH, step="match_notes")

    @staticmethod
    def match_notes_GOAT(track: "Track", delta: float) -> None:
        """
        Here, just the String assignment is switched. The string index can be found in GT,
        whereas in GuitarSet is is implied by the extraction of notes on the hex-signal.
        """
        gt_notes = [n for n in track.notes if n.origin == "gt"]
        pred_notes = [n for n in track.notes if n.origin == "model"]

        track.gt_notes = gt_notes  # ALL gt notes — matched ones will have match=True

        for pred in pred_notes:
            matched_gt = next(
                (gt for gt in gt_notes
                 if gt.attributes.midi_note == pred.attributes.midi_note
                 and gt.attributes.program == pred.attributes.program
                 and abs(pred.attributes.onset - gt.attributes.onset) <= delta),
                None,
            )
            if matched_gt:
                pred.valid = True
                pred.match = True
                pred.dataset = matched_gt.dataset
                pred.attributes.string_index = matched_gt.attributes.string_index
                matched_gt.match = True
            else:
                # matched_gt.invalidate(FilterReason.NO_MATCH, step="match_notes")
                pred.invalidate(FilterReason.NO_MATCH, step="match_notes")

    @staticmethod
    def match_notes_between_strings(
            track_audio,
            delta: float,
            notes: List[FeatureNote],
    ) -> None:
        """
        Resolve string-assignment conflicts among valid notes by energy comparison.
        The weaker duplicate is *invalidated* (automatically drops from valid_notes).
        """
        valid = [n for n in notes if n.valid]

        for i, pred in enumerate(valid):
            for pred2 in valid[i + 1:]:
                if not pred2.valid:  # might have been invalidated earlier
                    continue
                if pred.attributes.string_index == pred2.attributes.string_index:
                    continue
                if pred.attributes.program != pred2.attributes.program:
                    continue
                if pred.attributes.midi_note != pred2.attributes.midi_note:
                    continue
                if abs(pred.attributes.onset - pred2.attributes.onset) > delta:
                    continue

                sr = track_audio.sampling_rate

                def _energy(note: FeatureNote) -> float:
                    s = int(note.attributes.onset * sr)
                    e = int(note.attributes.offset * sr)
                    sig = pf.Signal(track_audio.time[note.attributes.string_index, s:e], sampling_rate=sr)
                    return pf.dsp.energy(sig)

                if _energy(pred) >= _energy(pred2):
                    pred2.invalidate(FilterReason.MISMATCH_BETWEEN_STRINGS,
                                     step="match_notes_between_strings")
                else:
                    pred.invalidate(FilterReason.MISMATCH_BETWEEN_STRINGS,
                                    step="match_notes_between_strings")