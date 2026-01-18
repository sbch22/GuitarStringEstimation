import os
import sys
sys.path.append(os.path.abspath(''))

from collections import defaultdict
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track, TrackAudio
import pyfar as pf
import glob
import re

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

    SECONDS_PER_TICK = 60.0 / (tempo * 3840)

    STANDARD_TUNING = {
        6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64
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

            open_midi = STANDARD_TUNING[string] + downtune
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
                    string_index=n["string"],
                    fret=n["fret"]
                )

                feature_notes.append(
                    FeatureNote(
                        attributes=attr,
                        features=Features(),
                        origin="dadagp"
                    )
                )

            active_notes.clear()
            current_tick = end_tick

        # ignore: new_measure, rest, nfx:hammer, nfx:tie, etc.

    return Track(
        name=track_id,
        notes=feature_notes,
        audio=TrackAudio(),
        metadata={
            "tempo": tempo,
            "downtune": downtune,
            "source": "GOAT-DadaGP"
        }
    )

def load_track_audio(track: Track, data_dir: str):
    base_name = track.name  # e.g. "item_0"

    paths = {
        "clean": os. path.join(data_dir, base_name + ".wav")
        # here implement for amp 1 - 6
    }

    for attr, file_path in paths.items():
        if os.path.exists(file_path):
            print(f"Loading {attr} from {file_path}")
            setattr(track.audio, attr, pf.io.read_audio(file_path))
        else:
            print(f"Missing {attr} file for {base_name}: {file_path}")


def preprocess_dataset(data_dir, save_dir):
    dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, 'item_*'))
        if os.path.isdir(d)
    ])


    for dir in dirs:
        # Convert all annotations to notes and note events
        GOAT_id = os.path.basename(dir)

        track_DadaGP = os.path.join(dir, f'{GOAT_id}.txt')

        if os.path.exists(track_DadaGP):
            track = create_track_from_DadaGP(track_DadaGP, GOAT_id)
            load_track_audio(track, dir) # load all needed audio files


            track_filename = GOAT_id + '.pkl'
            save_path = os.path.join(save_dir, track_filename)
            track.save(save_path)
            print(f'Saved track object: {save_path}')

def main():
    data_dir = '../../data/GOAT/data/'
    save_dir = '../noteData/GOAT/'
    preprocess_dataset(data_dir, save_dir)

# %%
if __name__ == "__main__":
    main()