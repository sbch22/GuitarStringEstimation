import os
import sys
sys.path.append(os.path.abspath(''))

import pickle
import argparse
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from configparser import ConfigParser

from utils.Track_dataclass import filter_analysis
from partialtracking import calculate_features


# ── Grid für GridSearch ──────────────────────────────────────────────────────
PARAM_GRID = {
    "svm__C":     [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto", 0.001, 0.01],
}


def build_pipeline():
    """SVM pipeline with fixed hyperparameters (C=10, gamma=0.001, rbf)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
        ("scaler",  StandardScaler()),
        ("svm",     SVC(kernel="rbf", C=10, gamma=0.001, probability=True)),
    ])


def load_data(config_train):
    """Loads all feature vectors and labels from .pkl track files."""
    track_directory = config_train.get("paths", "track_directory")
    audio_types_raw = config_train.get("paths", "audio_types")
    audio_types = [a.strip() for a in audio_types_raw.split(",")]

    filepaths = [
        os.path.join(track_directory, fn)
        for fn in os.listdir(track_directory)
        if os.path.isfile(os.path.join(track_directory, fn)) and fn.endswith(".pkl")
    ]

    all_feature_vectors, all_labels, all_notes = [], [], []
    sample_features = None

    for i, filepath in enumerate(filepaths, 1):
        print(f"\n[{i}/{len(filepaths)}] Processing {filepath}")
        with open(filepath, "rb") as f:
            track = pickle.load(f)

        for note in track.notes:
            all_notes.append(note)

        for audio_type in audio_types:
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

    FX     = np.stack(all_feature_vectors)
    labels = np.array(all_labels)
    return FX, labels, all_notes, sample_features


def nan_report(FX):
    nan_rate = np.isnan(FX).mean(axis=0)
    print(f"\nFeatures mit >50% NaN: {(nan_rate > 0.5).sum()} / {FX.shape[1]}")
    print(f"Features mit any NaN:  {(nan_rate > 0).sum()} / {FX.shape[1]}")


def run_fixed(FX, labels, output_path=None):
    """
    Mode 1 – Fixed hyperparameters (C=10, gamma=0.001, rbf).
    Trains on the full dataset.
    """
    print(f"\n[Fixed] Training on all {len(labels)} samples...")
    model = build_pipeline()
    model.fit(FX, labels)

    out = output_path or "svm_fixed.joblib"
    joblib.dump(model, out)
    print(f"[Fixed] Model saved: {out}")
    return model


def run_gridsearch(FX, labels, subset_frac=0.3, n_splits=5,
                   random_state=42, output_path=None):
    """
    Mode 2 – GridSearch on a random subset.
    Draws subset_frac % of data stratified, then runs GridSearchCV.
    """
    print(f"\n[GridSearch] Sampling {subset_frac*100:.0f}% of data "
          f"({int(len(labels)*subset_frac)} of {len(labels)} samples)...")

    FX_sub, _, labels_sub, _ = train_test_split(
        FX, labels,
        train_size=subset_frac,
        stratify=labels,
        random_state=random_state,
    )

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        build_pipeline(), PARAM_GRID,
        cv=cv, scoring="f1_macro", n_jobs=-1, verbose=2,
    )

    print(f"[GridSearch] Running GridSearchCV (CV={n_splits})...")
    grid.fit(FX_sub, labels_sub)

    print(f"\n[GridSearch] Best params:    {grid.best_params_}")
    print(f"[GridSearch] Best f1_macro:  {grid.best_score_:.3f}")

    out = output_path or f"svm_gridsearch_{int(subset_frac*100)}pct.joblib"
    joblib.dump(grid.best_estimator_, out)
    print(f"[GridSearch] Model saved: {out}")
    return grid.best_estimator_, grid.best_params_


def main():
    parser = argparse.ArgumentParser(description="SVM Training – GuitarSet")
    parser.add_argument(
        "--mode",
        choices=["fixed", "gridsearch"],
        default=None,
        help=(
            "'fixed'      – Fixed hyperparameters, trains on full dataset\n"
            "'gridsearch' – GridSearch on a random 30%% subset"
        ),
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=0.3,
        help="Only for --mode gridsearch. Fraction of data to use (0–1). Default: 0.3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the saved model (.joblib). Optional.",
    )
    args = parser.parse_args()

    # Interactive prompt if no mode was passed via CLI
    if args.mode is None:
        print("\nSelect training mode:")
        print("  [1] fixed      – Fixed hyperparameters, trains on full dataset")
        print("  [2] gridsearch – GridSearch on a random subset")
        choice = input("\nEnter 1 or 2: ").strip()
        args.mode = "fixed" if choice == "1" else "gridsearch"

    config_train = ConfigParser()
    config_train.read("configs/config_train_GuitarSet.ini")

    calculate_features.main(config_train)

    FX, labels, all_notes, sample_features = load_data(config_train)
    filter_analysis(all_notes)
    nan_report(FX)

    if args.mode == "fixed":
        run_fixed(FX, labels, output_path=args.output)
    elif args.mode == "gridsearch":
        run_gridsearch(FX, labels, subset_frac=args.subset, output_path=args.output)


if __name__ == "__main__":
    main()