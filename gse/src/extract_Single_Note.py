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
    data_dir = '../../data/Custom_Guitar_Dataset/single_notes/IDMT_SMT_GUITAR_V2_cut/dataset2/'
    save_dir = '../noteData/single_note/'

    audio_dir = os.path.join(data_dir, 'audio')

    # Collect all audio files and split
    all_audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    random.shuffle(all_audio_files)

    split_idx = int(len(all_audio_files) * 0.8)
    train_files = set(all_audio_files[:split_idx])
    test_files  = set(all_audio_files[split_idx:])

    # Create output directories
    train_dir = os.path.join(save_dir, 'train')
    test_dir  = os.path.join(save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
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
            dataset="single_note",
        )

        for event in root.findall('event'):
            pitch      = int(event.find('pitch').text)
            onset      = float(event.find('onsetSec').text)
            fret       = int(event.find('fretNumber').text)
            string_num = int(event.find('stringNumber').text)
            freq       = (440 / 32) * (2 ** ((pitch - 9) / 12))

            audio  = pf.io.read_audio(audio_filepath)
            offset = audio.signal_length

            attr = Attributes(
                midi_note=pitch,
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
        split_subdir = train_dir if audio_filepath in train_files else test_dir
        track_save_path = os.path.join(split_subdir, track.name)
        track.save(track_save_path + '.pkl')

# %%
if __name__ == "__main__":
    main()