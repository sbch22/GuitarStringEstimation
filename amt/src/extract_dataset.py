import os
import sys
sys.path.append(os.path.abspath(''))

import numpy as np
from typing import Tuple, Dict, Literal
import jams
from utils.note_event_dataclasses import Note, NoteEvent
import glob
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from collections import defaultdict


def extract_jams(jams_filename):
    annotation_directory = '../../data/guitarset_yourmt3_16k/annotation/'
    jams_filepath = os.path.join(annotation_directory, jams_filename)

    if os.path.exists(jams_filepath):
        print(f"Extracting GT from {jams_filepath}")
    else:
        print(f"Annotation file not found: {jams_filepath}")

    assert os.path.exists(jams_filepath)

    # load annotation
    jam = jams.load(jams_filepath)

    return jam



#Copied from preprocess_guitarset
def create_note_event_and_note_from_jam(jam_file: str, id: str) -> Tuple[Dict, Dict]:
    jam = jams.load(jam_file)

    notes = []
    contours = defaultdict(list)  # Speichert Pitch-Contour nach ihrem 'index'

    # Erstmal alle pitch_contour Daten sammeln
    for ann in jam.annotations:
        if ann.namespace == "pitch_contour":
            for obs in ann.data:
                if isinstance(obs.value, dict) and "frequency" in obs.value:
                    index = obs.value.get("index", None)  # Falls kein Index vorhanden, bleibt es None
                    contours[index].append((obs.time, obs.value["frequency"]))

    # Nun die MIDI-Noten extrahieren und mit der passenden Pitch-Contour verknüpfen
    for ann in jam.annotations:
        if ann.namespace == "note_midi":
            for obs in ann.data:
                if isinstance(obs.value, float):  # MIDI-Noten sind Floats
                    note_contour = []
                    for index, contour in contours.items():
                        # Nur die Werte übernehmen, die im Onset-Offset-Bereich der Note liegen
                        note_contour += [(t, f) for t, f in contour if obs.time <= t <= obs.time + obs.duration]

                    note = Note(is_drum=False,
                                program=24,
                                onset=obs.time,
                                offset=obs.time + obs.duration,
                                pitch=round(obs.value),
                                velocity=1,
                                contour=note_contour)
                    notes.append(note)



    # Sort, validate, and trim notes
    notes = sort_notes(notes)
    notes = validate_notes(notes)
    notes = trim_overlapping_notes(notes)


    return ({  # notes
        'guitarset_id': id,
        'program': [24],
        'is_drum': [0],
        'duration_sec': jam.file_metadata.duration,
        'notes': notes,
    }
    , {  # note_events
        'guitarset_id': id,
        'program': [24],
        'is_drum': [0],
        'duration_sec': jam.file_metadata.duration,
        'note_events': note2note_event(notes),
    })


def preprocess_dataset(base_dir):
    all_ann_files = glob.glob(os.path.join(base_dir, 'annotation/*.jams'), recursive=True)
    assert len(all_ann_files) == 360

    for ann_file in all_ann_files:
        # Convert all annotations to notes and note events
        guitarset_id = os.path.basename(ann_file).split('.')[0]
        notes, note_events = create_note_event_and_note_from_jam(ann_file, guitarset_id)

        notes_file = ann_file.replace('.jams', '_notes.npy')

        #TODO: this needs to be commented back in after debugging and finding of string annotation.

        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')




# Main
def main():
    base_dir = '../../data/guitarset_yourmt3_16k'
    preprocess_dataset(base_dir)



# %%
if __name__ == "__main__":
    main()