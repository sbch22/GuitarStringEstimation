import os
import sys
sys.path.append(os.path.abspath(''))
import jams
import glob
from collections import defaultdict
from FeatureNote_dataclass import FeatureNote, GT, Features
from Track_dataclass import Track

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

                gt = GT(
                    pitch=obs.value,
                    is_drum=False,
                    program=24,
                    onset=obs.time,
                    offset=obs.time + obs.duration,
                    velocity=1,
                    midi_note=round(obs.value),
                    contour=note_contour,
                )

                notes.append(FeatureNote(gt=gt, features=Features()))

    track = Track(
        name=track_id,
        notes=notes,
        audio=None,  # can load later
        sample_rate=None,
        metadata={"duration_sec": jam.file_metadata.duration, "source": "GuitarSet"},
    )

    # Apply preprocessing steps:
    track.notes = Track.sort_notes(track.notes)
    track.notes = Track.validate_notes(track.notes)
    track.notes = Track.trim_overlapping_notes(track.notes)

    return track


def preprocess_dataset(data_dir, save_dir):
    all_ann_files = glob.glob(os.path.join(data_dir, 'annotation/*.jams'), recursive=True)
    assert len(all_ann_files) == 360

    for ann_file in all_ann_files:
        # Convert all annotations to notes and note events
        guitarset_id = os.path.basename(ann_file).split('.')[0]

        track = create_track_from_jam(ann_file, guitarset_id)
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