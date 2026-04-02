import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(os.path.abspath(''))

import os
import pickle
import numpy as np
from configparser import ConfigParser
import joblib
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import calculate_features
from utils.Track_dataclass import filter_analysis

def layout_to_csv(layout: dict[str, int], csv_path: str = "feature_layout.csv") -> None:
    rows = []
    col = 0
    for segment, length in layout.items():
        for idx in range(length):
            rows.append({
                "feature_idx": col + idx,
                "segment":     segment,
                "position":    idx,
            })
        col += length

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Feature layout saved to {csv_path}  ({col} features total)")

def nan_analysis(FX: np.ndarray, layout: dict[str, int]) -> None:
    """
    Prints NaN rates per feature segment and – for array segments –
    per position (partial order) across all notes.

    Parameters
    ----------
    FX     : (n_notes, n_features) stacked feature matrix
    layout : {segment_name: length} as returned by Features.segment_layout()
    """
    n_notes = FX.shape[0]
    total_nan = np.isnan(FX).sum()
    print(f"\n{'═'*60}")
    print(f"NaN Analysis  –  {n_notes} notes, {FX.shape[1]} features total")
    print(f"Overall NaN rate: {total_nan}/{FX.shape[1]*n_notes} "
          f"= {total_nan / FX.size:.1%}")
    print(f"{'═'*60}")

    col = 0
    for name, length in layout.items():
        seg = FX[:, col : col + length]          # (n_notes, length)
        nan_mask = np.isnan(seg)
        total_seg_nan = nan_mask.sum()
        seg_nan_rate = total_seg_nan / seg.size

        print(f"\n▸ {name}  [{length} dim]  –  NaN rate: "
              f"{total_seg_nan}/{seg.size} = {seg_nan_rate:.1%}")

        # Per-order breakdown only makes sense for array segments
        if length > 1:
            per_order = nan_mask.mean(axis=0)    # fraction across notes per position
            # Only print if there's any NaN at all (keeps output clean)
            if per_order.any():
                print(f"  {'Idx':>4}  {'NaN-Rate':>10}  {'Bar':}")
                for idx, rate in enumerate(per_order):
                    bar = "█" * int(rate * 20)
                    print(f"  {idx:>4}  {rate:>9.1%}  {bar}")
            else:
                print("  → no NaNs in any position")

        col += length

    print(f"\n{'═'*60}\n")


def main(subset):
    # load configs
    if subset == 'GuitarSet':
        config_train = ConfigParser()
        config_train.read('configs/config_train_GuitarSet.ini')
    elif subset == 'IDMT_dataset3':
        config_train = ConfigParser()
        config_train.read('configs/config_train_IDMT.ini')
    elif subset == 'single_note_IDMT':
        config_train = ConfigParser()
        config_train.read('configs/config_train_single_note_IDMT.ini')

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
        and filename.endswith(".pkl")
        # and "solo" in filename
    ]

    all_feature_vectors = []
    all_labels = []
    all_notes = []
    sample_features = None


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

                if sample_features is None:
                    sample_features = note.features[audio_type]

    layout = sample_features.segment_layout()
    layout_to_csv(layout, "feature_layout.csv")

    FX = np.stack(all_feature_vectors)  # (total_notes, N)
    labels = np.array(all_labels)

    nan_analysis(FX, layout)
    filter_analysis(all_notes)


    # ── NaN analysis before imputation ───────────────────────────────────────
    nan_rate = np.isnan(FX).mean(axis=0)  # per-feature NaN rate
    print(f"\nFeatures with >50% NaN: {(nan_rate > 0.5).sum()} / {FX.shape[1]}")
    print(f"Features with any NaN:  {(nan_rate > 0).sum()} / {FX.shape[1]}")

    print("\nRunning SVM Fit.")
    # ── Pipeline ────────────────────────────────────────────────────────────────
    SVM = Pipeline([
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)), # add indicator saves if nan was encountered in indicator variable
        ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=0.95)),      # Feature-Reduktion
        ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True)),
    ])

    # # ── Grid Search
    # param_grid = {
    #     'svm__C': [0.1, 1, 10, 100],
    #     'svm__gamma': ['scale', 'auto', 0.001, 0.01]
    # }

    # # Und Daten auf 20% reduzieren um Laufzeit abzuschätzen
    # FX_sub, _, labels_sub, _ = train_test_split(
    #     FX, labels, train_size=0.3, stratify=labels, random_state=42
    # )

    # print("\nRunning GridSearchCV...")
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid = GridSearchCV(SVM, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    # grid.fit(FX_sub, labels_sub)

    # print(f"Best params:       {grid.best_params_}")
    # print(f"Best CV f1_macro:  {grid.best_score_:.3f}")
    #
    # best_model = grid.best_estimator_

    # Output
    # joblib.dump(best_model, "svm_pipeline_CV5_GS_20p_old.joblib")

    SVM.fit(FX, labels)
    joblib.dump(SVM, "SVM_full_post-DAGA_noGS.joblib") #TODO: wenn das nicht gut ist -> train on Solo
    # print(f"Best f1_macro:  {SVM.score:.3f}")



if __name__ == "__main__":
    # subset_single_note_IDMT = 'single_note_IDMT'
    subset_GuitarSet = 'GuitarSet'

    main(subset_GuitarSet)
