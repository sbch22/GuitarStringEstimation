import os
import sys
sys.path.append(os.path.abspath(''))

from collections import defaultdict
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track
import pyfar as pf
import glob
import re
import random

def create_track_from_DadaGP(DadaGP_file: str, track_id: str) -> Track:
    with open(DadaGP_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    tempo = 120
    downtune = 0

    # Parse global metadata
    for l in lines:
        if l.startswith("tempo:"):
            tempo = int(l.split(":")[1])
        if l.startswith("downtune:"):
            downtune = int(l.split(":")[1])

    downtune += 12  # Artefakt herausrechnen
    if downtune != 0:
        return None

    SECONDS_PER_TICK = 60.0 / (tempo * 3840)

    STANDARD_TUNING = {
        0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64
    }

    current_tick = 0
    active_notes = []
    feature_notes = []

    note_re = re.compile(r"note:s(\d):f(\d+)")

    for l in lines:
        # NOTE EVENTS
        if "note:" in l:
            m = note_re.search(l)
            if not m:
                continue

            string = int(m.group(1))
            fret = int(m.group(2))

            open_midi = STANDARD_TUNING[string - 1] + downtune
            midi = open_midi + fret
            pitch_hz = (440 / 32) * (2 ** ((midi - 9) / 12))

            active_notes.append({
                "string": string,
                "fret": fret,
                "midi": midi,
                "pitch": pitch_hz,
                "onset_tick": current_tick
            })

        # TIME ADVANCE
        elif l.startswith("wait:"):
            wait_ticks = int(l.split(":")[1])
            end_tick = current_tick + wait_ticks

            for n in active_notes:
                onset_sec = n["onset_tick"] * SECONDS_PER_TICK
                offset_sec = end_tick * SECONDS_PER_TICK

                attr = Attributes(
                    is_drum=False,
                    program=24,
                    onset=onset_sec,
                    offset=offset_sec,
                    velocity=1,
                    midi_note=n["midi"],
                    pitch=n["pitch"],
                    string_index=n["string"]-1, # -1 because 1-6 instead of 0-5
                    fret=n["fret"]
                )

                feature_notes.append(
                    FeatureNote(
                        attributes=attr,
                        valid=True,
                        features=Features(),
                        origin="gt",
                        dataset="GOAT"
                    )
                )

            active_notes.clear()
            current_tick = end_tick

    # ignore: new_measure, rest, nfx:hammer, nfx:tie, etc.
    return Track(
        name=track_id,
        notes=feature_notes,
        metadata={
            "tempo": tempo,
            "downtune": downtune,
            "source": "GOAT"
        }
    )


def load_track_audio_paths(track: Track, data_dir: str):
    base_name = track.name  # e.g. "item_0"

    paths = {
        "clean": os. path.join(data_dir, base_name + ".wav"),
        "amp1": os.path.join(data_dir, base_name + "_amp_1.wav"),
        "amp2": os.path.join(data_dir, base_name + "_amp_2.wav"),
        "amp3": os.path.join(data_dir, base_name + "_amp_3.wav"),
        "amp4": os.path.join(data_dir, base_name + "_amp_4.wav"),
        "amp5": os.path.join(data_dir, base_name + "_amp_5.wav")
    }

    for attr, file_path in paths.items():
        if os.path.exists(file_path):
            print(f"Loading {attr} from {file_path}")
            track.audio_paths = paths



def preprocess_dataset(data_dir, save_dir):
    dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, 'item_*'))
        if os.path.isdir(d)
    ])

    # Collect all valid tracks first
    valid_dirs = []
    for dir in dirs:
        GOAT_id = os.path.basename(dir)
        track_DadaGP = os.path.join(dir, f'{GOAT_id}.txt')
        if os.path.exists(track_DadaGP):
            valid_dirs.append((dir, GOAT_id))

    # Split into test (32 random) and train (rest)
    random.shuffle(valid_dirs)
    test_dirs = set(d[0] for d in valid_dirs[:32])

    # Create subdirectories
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for dir, GOAT_id in valid_dirs:
        track_DadaGP = os.path.join(dir, f'{GOAT_id}.txt')

        track = create_track_from_DadaGP(track_DadaGP, GOAT_id)
        if track is None:
            print(f"Skipping {GOAT_id} (downtuned)")
            continue

        load_track_audio_paths(track, dir)

        split = 'test' if dir in test_dirs else 'train'
        save_path = os.path.join(save_dir, split, GOAT_id + '.pkl')
        track.save(save_path)
        print(f'[{split}] Saved track object: {save_path}')

def main():
    data_dir = '../../data/GOAT/data/'
    save_dir = '../noteData/GOAT/'
    preprocess_dataset(data_dir, save_dir)

# %%
if __name__ == "__main__":
    main()