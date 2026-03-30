import os
import sys
sys.path.append(os.path.abspath(''))
import glob
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track
import pyfar as pf
import xml.etree.ElementTree as ET
import random

def main():
    data_dir = '../../data/IDMT/dataset3/'
    save_dir = '../noteData/IDMT_dataset3/'

    audio_dir = os.path.join(data_dir, 'audio')

    # Collect all audio files and split
    all_audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))

    # Create output directories
    test_dir  = os.path.join(save_dir, 'test')
    os.makedirs(test_dir,  exist_ok=True)

    for audio_filepath in all_audio_files:
        ann_filename = audio_filepath.replace('.wav', '.xml')
        ann_filepath = ann_filename.replace('/audio/', '/annotation/')
        tree = ET.parse(ann_filepath)
        root = tree.getroot()

        filename = os.path.basename(audio_filepath)

        track = Track(
            name=filename.split('.')[0],
            audio_paths={"mono": audio_filepath},
            dataset="IDMT-dataset3",
        )

        for event in root.findall('transcription/event'):
            midi_note      = int(event.find('pitch').text)
            onset      = float(event.find('onsetSec').text)
            offset     = float(event.find('offsetSec').text)
            fret       = int(event.find('fretNumber').text)
            string_num = int(event.find('stringNumber').text)
            freq       = (440 / 32) * (2 ** ((midi_note - 9) / 12))


            attr = Attributes(
                midi_note=midi_note,
                is_drum=False,
                program=24,
                onset=onset,
                offset=offset,
                pitch=freq,
                string_index=string_num - 1,
                fret=fret,
            )

            note = FeatureNote(
                origin='single_note',
                match=True,
                attributes=attr,
                valid=True,
            )
            track.notes.append(note)

        # Save to train or test directory
        track_save_path = os.path.join(test_dir, track.name)
        track.save(track_save_path + '.pkl')

# %%
if __name__ == "__main__":
    main()