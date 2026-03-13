import os
import sys
sys.path.append(os.path.abspath(''))
import glob
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track, TrackAudio
import pyfar as pf
import xml.etree.ElementTree as ET


def main():
    data_dir = '../../data/Custom_Guitar_Dataset/single_notes/IDMT_SMT_GUITAR_V2_cut/dataset2/'
    save_dir = '../noteData/single_note/dev/'

    audio_dir = os.path.join(data_dir, 'audio')

    for audio_filepath in glob.glob(os.path.join(audio_dir, '*.wav')):
        audio = pf.io.read_audio(audio_filepath)

        ann_filename = audio_filepath.replace('.wav', '.xml')
        ann_filepath = ann_filename.replace('/audio/', '/annotation/')
        tree = ET.parse(ann_filepath)
        root = tree.getroot()

        filename = os.path.basename(audio_filepath)

        track = Track(
            name=filename.split('.')[0],
            audio=TrackAudio(
                mono_mic=audio,
            )
        )

        for event in root.findall('event'):
            pitch = int(event.find('pitch').text)
            onset = float(event.find('onsetSec').text)
            fret = int(event.find('fretNumber').text)
            string_num = int(event.find('stringNumber').text)
            freq = (440 / 32) * (2 ** ((pitch - 9) / 12))

            # create attributes
            attr = Attributes(
                midi_note=pitch,
                is_drum=False,
                program=24,
                onset=onset,
                pitch=freq,
                string_index=string_num,
                fret=fret,
            )

            # create Note
            note = FeatureNote(
                origin='single_note',
                match=True,
                attributes=attr,
            )
            track.notes.append(note)

        track_save_path = os.path.join(save_dir, track.name)
        track.save(track_save_path+'.pkl')

# %%
if __name__ == "__main__":
    main()