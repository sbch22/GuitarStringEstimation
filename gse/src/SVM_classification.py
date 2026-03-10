import os
import sys
sys.path.append(os.path.abspath(''))

import os
import pickle
import numpy as np
from configparser import ConfigParser
import joblib


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline   # capital P, from pipeline module
from sklearn.svm import SVC             # capital SVC, from svm module



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

    all_feature_vectors = []
    all_labels = []

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Processing {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)

        valid_notes = [note for note in track.notes if
                       note.valid == True and note.origin == 'gt']

        for note in valid_notes:
            all_feature_vectors.append(note.features.feature_vector)
            all_labels.append(note.attributes.string_index)  # GT






if __name__ == "__main__":
    config = ConfigParser()
    config.read('config_train.ini')
    main(config)