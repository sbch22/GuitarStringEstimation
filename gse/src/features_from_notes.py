import os
import sys
sys.path.append(os.path.abspath(''))

import multiprocessing as mp
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.FeatureNote_dataclass
import utils.Track_dataclass
from find_partials import filter_analysis

from configparser import ConfigParser



def note_number_of_partials(note):




def main(config):
    W = config.getint('train', 'W')
    H = config.getint('train', 'H')
    beta_max = config.getfloat('train', 'beta_max')
    threshold = config.getint('train', 'threshold')
    plot = config.getboolean('train', 'plot')
    track_directory = config.get('paths', 'track_directory')

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    feature_matrix =


    # Process files one by one (DEBUG FRIENDLY)
    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Processing {filepath}")
        with open(filepath, "rb") as f:
            track = pickle.load(f)

            # TODO: calculate track-features




            track.save(filepath)
            print(f"Saved {filepath}")

            return f"Success: {os.path.basename(filepath)}"



if __name__ == '__main__':
    config = ConfigParser()
    config.read('config_train.ini')

    main(config)