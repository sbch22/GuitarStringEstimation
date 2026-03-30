import os
import sys
sys.path.append(os.path.abspath(''))

import pickle
import numpy as np
import soundfile as sf
from configparser import ConfigParser
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_STRINGS = list(range(6))          # strings 0–5
TARGET_FRETS   = [5, 12]                 # frets of interest
NOTES_PER_CASE = 20                      # first N notes per (string, fret)
OUTPUT_DIR     = "../../data/GuitarSet/note_audio_clips"

# How much audio to capture around each note (seconds).
# If your Note objects carry an `offset` attribute, that is used instead.
FALLBACK_DURATION_S = 0.5               # used only when no offset is available
PRE_ONSET_PAD_S     = 0.01              # small lead-in before the onset


def get_note_offset(note, fallback_duration: float) -> float:
    """Return the note's end time in seconds."""
    attrs = note.attributes
    # Try common attribute names for note end / duration
    for attr in ("offset", "end", "end_time", "offset_time"):
        if hasattr(attrs, attr):
            val = getattr(attrs, attr)
            if val is not None:
                return float(val)
    if hasattr(attrs, "duration") and attrs.duration is not None:
        return float(attrs.onset) + float(attrs.duration)
    # Fall back to a fixed window after onset
    return float(attrs.onset) + fallback_duration


def extract_clip(audio: np.ndarray, sr: int,
                 onset_s: float, offset_s: float) -> np.ndarray:
    """Slice [onset - pad, offset] from *audio* (1-D or 2-D, samples × channels)."""
    start = max(0, int((onset_s - PRE_ONSET_PAD_S) * sr))
    end   = min(len(audio), int(offset_s * sr))
    return audio[start:end]


def main(subset: str):
    config_test = ConfigParser()
    if subset == "comp":
        config_test.read("configs/config_test_comp.ini")
    elif subset == "solo":
        config_test.read("configs/config_test_solo.ini")

    track_directory = config_test.get("paths", "track_directory")

    filepaths = sorted([
        os.path.join(track_directory, fn)
        for fn in os.listdir(track_directory)
        if fn.endswith(".pkl") and os.path.isfile(os.path.join(track_directory, fn))
    ])

    # ── Accumulator: (string, fret) → list of (onset_s, offset_s, audio_path) ──
    # We collect across *all* tracks until each bucket has NOTES_PER_CASE entries.
    buckets: dict[tuple[int,int], list[tuple[float, float, str]]] = defaultdict(list)
    full_buckets: set[tuple[int,int]] = set()
    needed = {(s, f) for s in TARGET_STRINGS for f in TARGET_FRETS}

    for i, filepath in enumerate(filepaths, 1):
        if full_buckets == needed:
            break  # all 12 buckets are full – no need to read more tracks

        print(f"[{i}/{len(filepaths)}] Scanning {filepath}")

        with open(filepath, "rb") as fh:
            track = pickle.load(fh)

        audio_path = track.audio_paths["mono_mic"]

        # Sort notes by onset so we get the chronologically first ones
        notes = sorted(track.valid_notes, key=lambda n: n.attributes.onset)

        for note in notes:
            attrs = note.attributes

            # ── Resolve string & fret ──────────────────────────────────────
            # Adjust these attribute names to match your Note class.
            string_idx = None
            fret_val   = None
            for sattr in ("string", "string_index", "string_num"):
                if hasattr(attrs, sattr):
                    string_idx = int(getattr(attrs, sattr))
                    break
            for fattr in ("fret", "fret_number", "fret_num"):
                if hasattr(attrs, fattr):
                    fret_val = int(getattr(attrs, fattr))
                    break

            if string_idx is None or fret_val is None:
                continue  # skip notes without string/fret info

            key = (string_idx, fret_val)
            if key not in needed or key in full_buckets:
                continue

            onset_s  = float(attrs.onset)
            offset_s = get_note_offset(note, FALLBACK_DURATION_S)

            buckets[key].append((onset_s, offset_s, audio_path))

            if len(buckets[key]) >= NOTES_PER_CASE:
                full_buckets.add(key)

    # ── Write WAV clips ───────────────────────────────────────────────────────
    print(f"\nWriting clips to {OUTPUT_DIR} …")

    for (string_idx, fret_val), entries in sorted(buckets.items()):
        subdir = os.path.join(OUTPUT_DIR, f"string{string_idx}_fret{fret_val:02d}")
        os.makedirs(subdir, exist_ok=True)

        # Cache open audio files to avoid re-reading the same file repeatedly
        cached_audio: dict[str, tuple[np.ndarray, int]] = {}

        for note_idx, (onset_s, offset_s, audio_path) in enumerate(entries[:NOTES_PER_CASE], 1):
            if audio_path not in cached_audio:
                audio, sr = sf.read(audio_path, always_2d=False)
                cached_audio[audio_path] = (audio, sr)
            audio, sr = cached_audio[audio_path]

            clip = extract_clip(audio, sr, onset_s, offset_s)
            if len(clip) == 0:
                print(f"  ⚠  Empty clip for string {string_idx} fret {fret_val} note {note_idx} – skipping")
                continue

            out_name = f"note_{note_idx:02d}_onset{onset_s:.3f}s.wav"
            out_path = os.path.join(subdir, out_name)
            sf.write(out_path, clip, sr)

        print(f"  string {string_idx} fret {fret_val:2d}: {min(len(entries), NOTES_PER_CASE)} clips → {subdir}")

    missing = needed - set(buckets.keys()) | {k for k, v in buckets.items() if len(v) < NOTES_PER_CASE}
    if missing:
        print("\n⚠  The following (string, fret) combinations had fewer than"
              f" {NOTES_PER_CASE} notes across all tracks:")
        for s, f in sorted(missing):
            found = len(buckets.get((s, f), []))
            print(f"    string {s} fret {f}: {found} note(s) found")


if __name__ == "__main__":
    main("solo")
