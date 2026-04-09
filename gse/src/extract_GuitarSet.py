import os
import sys
sys.path.append(os.path.abspath(''))
import jams
import glob
from collections import defaultdict
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track
import csv

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

                noteMIDI = round(obs.value)
                noteFREQ = (440 / 32) * (2 ** ((noteMIDI - 9) / 12))

                attr = Attributes(
                    is_drum=False,
                    program=24,
                    onset=obs.time,
                    offset=obs.time + obs.duration,
                    velocity=1,
                    midi_note= noteMIDI,
                    contour=note_contour,
                    pitch=noteFREQ,  # convert to frequency
                )

                notes.append(FeatureNote(
                    attributes=attr,
                    features=Features(),
                    origin='gt',
                    valid=True,
                    dataset="GuitarSet",
                ))

    track = Track(
        name=track_id,
        notes=notes,
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

    # load Senva train file list
    with open('../configs/trainSet.csv', newline='') as csvfile:
        train_filename_list = []
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            path_csv = row[:]
            filename = path_csv.pop()
            train_filename_list.append(filename)

            # also load solo file
            filename_solo = filename.replace("comp", "solo")
            train_filename_list.append(filename_solo)

    # load Senva test file list
    with open('../configs/testSet.csv', newline='') as csvfile:
        test_filename_list_comp = []
        test_filename_list_solo = []
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            path_csv = row[:]
            filename = path_csv.pop()
            test_filename_list_comp.append(filename)

            # also load solo file
            filename_solo = filename.replace("comp", "solo")
            test_filename_list_solo.append(filename_solo)

            # remove data-leakage: remove from solo from train
            if filename_solo in train_filename_list:
                train_filename_list.remove(filename)

    # ── collect split info for CSV export ──────────────────────────────────
    split_rows = []   # list of dicts: {split, subset, track_name, mono_mic_path}

    for test_filename in test_filename_list_comp:
        filebase = test_filename.split(".wav")[0]
        guitarset_id = filebase.split("_hex")[0]
        ann_filename = os.path.join(data_dir, 'annotation', guitarset_id + ".jams")

        track = create_track_from_jam(ann_filename, guitarset_id)

        base_name = track.name
        paths = {
            "mono_mic": os.path.join(data_dir, "audio_mono-mic", f"{base_name}_mic.wav"),
            "hex_debleeded": os.path.join(data_dir, "audio_hex-pickup_debleeded", f"{base_name}_hex_cln.wav"),
            "hex_mono_mix": os.path.join(data_dir, "audio_mono-pickup_mix", f"{base_name}_mix.wav"),
            "hex": os.path.join(data_dir, "audio_hex-pickup_original", f"{base_name}_hex.wav"),
        }
        track.audio_paths = paths

        track_filename = os.path.basename(ann_filename).replace('.jams', '_track.pkl')
        save_path = os.path.join(save_dir, 'GuitarSet', 'test', 'comp', track_filename)
        track.save(save_path)
        print(f'Saved track object: {save_path}')

        split_rows.append({"split": "test", "subset": "comp",
                           "track_name": base_name, "mono_mic_path": paths["mono_mic"]})

    for test_filename in test_filename_list_solo:
        filebase = test_filename.split(".wav")[0]
        guitarset_id = filebase.split("_hex")[0]
        ann_filename = os.path.join(data_dir, 'annotation', guitarset_id + ".jams")

        track = create_track_from_jam(ann_filename, guitarset_id)

        base_name = track.name
        paths = {
            "mono_mic": os.path.join(data_dir, "audio_mono-mic", f"{base_name}_mic.wav"),
            "hex_debleeded": os.path.join(data_dir, "audio_hex-pickup_debleeded", f"{base_name}_hex_cln.wav"),
            "hex_mono_mix": os.path.join(data_dir, "audio_mono-pickup_mix", f"{base_name}_mix.wav"),
            "hex": os.path.join(data_dir, "audio_hex-pickup_original", f"{base_name}_hex.wav"),
        }
        track.audio_paths = paths

        track_filename = os.path.basename(ann_filename).replace('.jams', '_track.pkl')
        save_path = os.path.join(save_dir, 'GuitarSet', 'test', 'solo', track_filename)
        track.save(save_path)
        print(f'Saved track object: {save_path}')

        split_rows.append({"split": "test", "subset": "solo",
                           "track_name": base_name, "mono_mic_path": paths["mono_mic"]})

    for train_filename in train_filename_list:
        filebase = train_filename.split(".wav")[0]
        guitarset_id = filebase.split("_hex")[0]
        ann_filename = os.path.join(data_dir, 'annotation', guitarset_id + ".jams")

        track = create_track_from_jam(ann_filename, guitarset_id)

        base_name = track.name
        paths = {
            "mono_mic": os.path.join(data_dir, "audio_mono-mic", f"{base_name}_mic.wav"),
            "hex_debleeded": os.path.join(data_dir, "audio_hex-pickup_debleeded", f"{base_name}_hex_cln.wav"),
            "hex_mono_mix": os.path.join(data_dir, "audio_mono-pickup_mix", f"{base_name}_mix.wav"),
            "hex": os.path.join(data_dir, "audio_hex-pickup_original", f"{base_name}_hex.wav"),
        }
        track.audio_paths = paths

        track_filename = os.path.basename(ann_filename).replace('.jams', '_track.pkl')
        save_path = os.path.join(save_dir, 'GuitarSet', 'train', track_filename)
        track.save(save_path)
        print(f'Saved track object: {save_path}')

        subset = "comp" if "comp" in base_name else "solo"
        split_rows.append({"split": "train", "subset": subset,
                           "track_name": base_name, "mono_mic_path": paths["mono_mic"]})

    # ── write split CSV ─────────────────────────────────────────────────────
    split_csv_path = os.path.join(save_dir, 'GuitarSet', 'split.csv')
    os.makedirs(os.path.dirname(split_csv_path), exist_ok=True)
    with open(split_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["split", "subset", "track_name", "mono_mic_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    print(f'Saved split CSV: {split_csv_path}')

# Main
def main():
    data_dir = '../../data/GuitarSet' # GuitarSet directory
    save_dir = '../noteData/' # save directory -> this saves all track & note objects
    preprocess_dataset(data_dir, save_dir)

# %%
if __name__ == "__main__":
    main()