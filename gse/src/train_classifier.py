import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(os.path.abspath(''))

import os
import pickle
import numpy as np
from configparser import ConfigParser
import joblib


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline   # capital P, from pipeline module
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier  # scales to millions
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate

import calculate_features
import find_partials
from utils.Track_dataclass import filter_analysis


def main():
    # load configs
    config_test = ConfigParser()
    config_test.read('config_test_comp.ini')

    config_train = ConfigParser()
    config_train.read('config_train.ini')

    # --- Step 1: Find partials ---
    find_partials.main(config_train)

    # --- Step 2: Calculate features ---
    calculate_features.main(config_train)

    track_directory = config_train.get('paths', 'track_directory')
    audio_types_raw = config_train.get('paths', 'audio_types')
    audio_types = [a.strip() for a in audio_types_raw.split(',')]

    # Collect all file paths
    filepaths = [
        os.path.join(track_directory, filename)
        for filename in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, filename))
    ]

    all_feature_vectors = []
    all_labels = []
    all_notes = []

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Processing {filepath}")

        with open(filepath, "rb") as f:
            track = pickle.load(f)

        for note in track.notes:
            all_notes.append(note)

        for audio_type in audio_types:
            # append all notes with filter reasons

            for note in track.valid_notes:
                if audio_type not in note.features:
                    continue
                fv = note.features[audio_type].feature_vector
                if fv is None:
                    continue
                all_feature_vectors.append(fv)
                all_labels.append(note.attributes.string_index)

    filter_analysis(all_notes)

    # stack over all tracks
    FX = np.stack(all_feature_vectors)  # (total_notes, N)
    labels = np.array(all_labels)  # (total_notes,)

    np.save("FX", FX, allow_pickle=True) # uncleaned with nans
    np.save("labels", labels, allow_pickle=True) # uncleaned with nans

    # ── Pipeline ────────────────────────────────────────────────────────────────
    SVM = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True))
        # Alternatives if training is too slow at 30k samples:
        # ('svm', LinearSVC(C=1.0, max_iter=2000))               # ~10-50x faster, linear only
        # ('svm', SGDClassifier(loss='hinge', max_iter=1000))     # fastest, needs more tuning
    ])

    # ── Quick CV estimate before full grid search ────────────────────────────────
    print("Running cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        SVM, FX, labels,
        cv=cv,
        scoring=['accuracy', 'f1_macro'],
        n_jobs=-1  # ← parallelizes the 5 folds across all cores
    )
    print(f"CV Accuracy:  {results['test_accuracy'].mean():.3f} ± {results['test_accuracy'].std():.3f}")
    print(f"CV F1 (macro): {results['test_f1_macro'].mean():.3f} ± {results['test_f1_macro'].std():.3f}")

    # ── Grid Search ─────────────────────────────────────────────────────────────
    # param_grid = {
    #     'svm__C': [0.1, 1, 10, 100],
    #     'svm__gamma': ['scale', 'auto', 0.001, 0.01]
    # }

    # print("\nRunning GridSearchCV...")
    # grid = GridSearchCV(
    #     SVM,
    #     param_grid,
    #     cv=cv,
    #     scoring='accuracy',
    #     n_jobs=-1,  # ← parallelizes all 4×4×5 = 80 fits across all cores
    #     verbose=2  # shows progress
    # )
    # grid.fit(FX, labels)

    # print(f"Best params: {grid.best_params_}")
    # print(f"Best CV accuracy: {grid.best_score_:.3f}")
    #
    # # Das beste Modell ist grid.best_estimator_ — das speicherst du
    # joblib.dump(grid.best_estimator_, "svm_pipeline.joblib")

    # Final fit on all data
    SVM.fit(FX, labels)
    joblib.dump(SVM, "svm_pipeline.joblib")






if __name__ == "__main__":
    main()