import os
import sys
sys.path.append(os.path.abspath(''))

import pickle
import csv
from configparser import ConfigParser
import numpy as np

def main(subset):
    # Load configs
    config_test = ConfigParser()
    if subset == 'comp':
        config_test.read('configs/config_test_comp.ini')
    elif subset == 'solo':
        config_test.read('configs/config_test_solo.ini')

    track_directory       = config_test.get('paths', 'track_directory')
    audio_types_raw       = config_test.get('paths', 'audio_types')
    audio_types           = [a.strip() for a in audio_types_raw.split(',')]
    transcription_directory = '../../data/GuitarSet/hjerrild_transcription/audio_mono-mic/martin/'

    # Collect all track pkl files
    filepaths = sorted([
        os.path.join(track_directory, fn)
        for fn in os.listdir(track_directory)
        if fn.endswith(".pkl") and os.path.isfile(os.path.join(track_directory, fn))
    ])

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Evaluating: {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)


        # TODO: Evaluate if model_note is actually good?
        model_notes = [n for n in track.notes if n.origin == 'model']

        onsets = [note.attributes.onset for note in model_notes]
        onsets.sort()
        # Build the matching CSV filename
        filename = os.path.basename(filepath)
        filename = os.path.splitext(filename)[0]  # strip .pkl
        filename = filename.replace('_track', '.csv')  # e.g. 00_BN1-129-Eb_comp.csv
        onsets_filepath = os.path.join('../../data/GuitarSet/hjerrild_transcription/audio_mono-mic/Onsets/', filename)

        os.makedirs(transcription_directory, exist_ok=True)

        with open(onsets_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['onset_s'])  # single header — easy to skip in MATLAB
            for onset in onsets:
                writer.writerow([round(float(onset), 6)])

        print(f"  Wrote {len(onsets)} onsets → {onsets_filepath}")



if __name__ == "__main__":
    main('solo')
    main('comp')