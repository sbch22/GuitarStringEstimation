import os
import sys
sys.path.append(os.path.abspath(''))
import jams
import glob
from collections import defaultdict
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track, TrackAudio
import pyfar as pf

def create_track_from_jam(jam_file: str, track_id: str) -> Track:
    jam = jams.load(jam_file)
    notes = []

    contours = defaultdict(list)  # Speichert Pitch-Contour nach ihrem 'index'
    # collect contours
    for ann in jam.annotations:
        if ann.namespace == "pitch_contour":
            for obs in ann.data:
                if isinstance(obs.value, dict) and "frequency" in obs.value:
                    index = obs.value.get("index", None)  # Falls kein Index vorhanden, bleibt es None
                    contours[index].append((obs.time, obs.value["frequency"]))

    # connect contour with midi note
    for ann in jam.annotations:
        if ann.namespace == "note_midi":
            for obs in ann.data:
                note_contour = []
                for index, contour in contours.items():
                    note_contour += [(t, f) for t, f in contour if obs.time <= t <= obs.time + obs.duration]

                attr = Attributes(
                    pitch= 440.0 * (2 ** ((obs.value - 69) / 12)), # convert to frequency
                    is_drum=False,
                    program=24,
                    onset=obs.time,
                    offset=obs.time + obs.duration,
                    velocity=1,
                    midi_note=obs.value,
                    contour=note_contour,
                )

                notes.append(FeatureNote(attributes=attr, features=Features(), origin='gt'))

    track = Track(
        name=track_id,
        notes=notes,
        audio=TrackAudio(),  # can load later
        metadata={"duration_sec": jam.file_metadata.duration, "source": "GuitarSet"},
    )

    # Apply preprocessing steps:
    track.notes = Track.sort_notes(track.notes)
    track.notes = Track.validate_notes(track.notes)
    track.notes = Track.trim_overlapping_notes(track.notes)

    return track

def load_track_audio(track: Track, data_dir: str):
    base_name = track.name  # e.g. "00_BN1-129-Eb_comp"

    paths = {
        "mono_mic": os.path.join(data_dir, "audio_mono-mic", f"{base_name}_mic.wav"),
        "hex_debleeded": os.path.join(data_dir, "audio_hex-pickup_debleeded", f"{base_name}_hex_cln.wav"),
        "hex_mono_mix": os.path.join(data_dir, "audio_mono-pickup_mix", f"{base_name}_mix.wav"),
    }

    for attr, file_path in paths.items():
        if os.path.exists(file_path):
            print(f"Loading {attr} from {file_path}")
            setattr(track.audio, attr, pf.io.read_audio(file_path))
        else:
            print(f"Missing {attr} file for {base_name}: {file_path}")

def preprocess_dataset(data_dir, save_dir):
    all_ann_files = glob.glob(os.path.join(data_dir, 'annotation/*.jams'), recursive=True)
    assert len(all_ann_files) == 360

    for ann_file in all_ann_files:
        # Convert all annotations to notes and note events
        guitarset_id = os.path.basename(ann_file).split('.')[0]

        track = create_track_from_jam(ann_file, guitarset_id)
        load_track_audio(track, data_dir) # load all needed audio files

        print(Track.__module__)

        track_filename = os.path.basename(ann_file).replace('.jams', '_track.pkl')
        save_path = os.path.join(save_dir, track_filename)
        track.save(save_path)
        print(f'Saved track object: {save_path}')

# Main
def main():
    data_dir = '../../data/guitarset_yourmt3_16k'
    save_dir = '../noteData/'
    preprocess_dataset(data_dir, save_dir)

# %%
if __name__ == "__main__":
    main()