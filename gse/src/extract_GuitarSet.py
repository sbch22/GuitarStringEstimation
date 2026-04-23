import sys
from pathlib import Path

# Resolve project root relative to this file and add to path — works regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jams
import csv
from collections import defaultdict

from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, Features
from gse.src.utils.Track_dataclass import Track

# Resolve config directory relative to this file
_SCRIPT_DIR = Path(__file__).resolve().parent


def create_track_from_jam(jam_file: Path, track_id: str) -> Track:
    """Convert JAMS file to Track object."""
    jam = jams.load(str(jam_file))
    notes = []
    contours = defaultdict(list)

    # Collect pitch contours
    for ann in jam.annotations:
        if ann.namespace == "pitch_contour":
            for obs in ann.data:
                if isinstance(obs.value, dict) and "frequency" in obs.value:
                    idx = obs.value.get("index")
                    contours[idx].append((obs.time, obs.value["frequency"]))

    # Map contours to MIDI notes
    for ann in jam.annotations:
        if ann.namespace == "note_midi":
            for obs in ann.data:
                note_contour = [
                    (t, f)
                    for contour in contours.values()
                    for t, f in contour
                    if obs.time <= t <= obs.time + obs.duration
                ]

                midi = round(obs.value)
                freq = (440 / 32) * (2 ** ((midi - 9) / 12))

                attr = Attributes(
                    is_drum=False,
                    program=24,
                    onset=obs.time,
                    offset=obs.time + obs.duration,
                    velocity=1,
                    midi_note=midi,
                    contour=note_contour,
                    pitch=freq,
                )

                notes.append(
                    FeatureNote(
                        attributes=attr,
                        features=Features(),
                        origin="gt",
                        valid=True,
                        dataset="GuitarSet",
                    )
                )

    track = Track(
        name=track_id,
        notes=notes,
        metadata={
            "duration_sec": jam.file_metadata.duration,
            "source": "GuitarSet",
        },
    )

    # Post-processing
    track.notes = Track.sort_notes(track.notes)
    track.notes = Track.validate_notes(track.notes)
    track.notes = Track.trim_overlapping_notes(track.notes)

    return track


def load_csv_list(path: Path, add_solo: bool = False) -> list:
    """Load filenames from CSV, optionally add solo versions."""
    filenames = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[-1]
            filenames.append(filename)
            if add_solo:
                filenames.append(filename.replace("comp", "solo"))
    return filenames


def build_audio_paths(base_name: str, data_dir: Path) -> dict:
    """Create audio path dictionary."""
    return {
        "mono_mic":      str(data_dir / "audio_mono-mic"                / f"{base_name}_mic.wav"),
        "hex_debleeded": str(data_dir / "audio_hex-pickup_debleeded"    / f"{base_name}_hex_cln.wav"),
        "hex_mono_mix":  str(data_dir / "audio_mono-pickup_mix"         / f"{base_name}_mix.wav"),
        "hex":           str(data_dir / "audio_hex-pickup_original"     / f"{base_name}_hex.wav"),
    }


def process_and_save(track_id: str, ann_path: Path, save_path: Path, data_dir: Path) -> Track:
    """Create track, attach audio paths, and save."""
    track = create_track_from_jam(ann_path, track_id)
    track.audio_paths = build_audio_paths(track.name, data_dir)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    track.save(str(save_path))
    print(f"Saved: {save_path}")

    return track


def preprocess_dataset(data_dir: Path, save_dir: Path) -> None:
    """Preprocess GuitarSet dataset and save tracks + split CSV."""
    ann_files = list((data_dir / "annotation").glob("*.jams"))
    assert len(ann_files) == 360, f"Expected 360 annotation files, found {len(ann_files)}"

    train_list = load_csv_list(_SCRIPT_DIR / "configs" / "trainSet.csv", add_solo=True)
    test_comp  = load_csv_list(_SCRIPT_DIR / "configs" / "testSet.csv")
    test_solo  = [f.replace("comp", "solo") for f in test_comp]

    # Remove leakage
    for f in test_solo:
        if f in train_list:
            train_list.remove(f)

    split_rows = []

    def handle_split(file_list: list, split: str, subset: str | None = None) -> None:
        for filename in file_list:
            filebase  = Path(filename).stem          # strips .wav if present
            track_id  = filebase.split("_hex")[0]
            ann_path  = data_dir / "annotation" / f"{track_id}.jams"

            track_filename = f"{track_id}_track.pkl"
            save_path = (
                save_dir / split / subset / track_filename
                if subset
                else save_dir / split / track_filename
            )

            track = process_and_save(track_id, ann_path, save_path, data_dir)

            split_rows.append({
                "split":         split,
                "subset":        subset if subset else ("comp" if "comp" in track.name else "solo"),
                "track_name":    track.name,
                "mono_mic_path": track.audio_paths["mono_mic"],
            })

    # Process splits
    handle_split(test_comp,  "test",  "comp")
    handle_split(test_solo,  "test",  "solo")
    handle_split(train_list, "train")

    # Save split CSV
    split_csv = save_dir / "split.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)

    with split_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "subset", "track_name", "mono_mic_path"])
        writer.writeheader()
        writer.writerows(split_rows)

    print(f"Saved split CSV: {split_csv}")


def main() -> None:
    """Entry point."""
    data_dir = PROJECT_ROOT / "data" / "GuitarSet"
    save_dir = PROJECT_ROOT / "data" / "GuitarSet" / "noteData"
    preprocess_dataset(data_dir, save_dir)


if __name__ == "__main__":
    main()