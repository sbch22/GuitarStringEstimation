# Guitar String Estimation on GuitarSet via YourMT3

> Reproducible code for the DAGA 2026 paper:
> **"Bestimmung gespielter Gitarrensaiten anhand von Audiofeatures basierend auf multipler Grundfrequenzschätzung"**
> ("Estimation of Played Guitar Strings Using Audio Features Based on Multiple Fundamental Frequency Estimation")
> — Simon Büchner, Paul A. Bereuter, Alois Sontacchi · IEM Graz

## Description

This project builds on **YourMT3** to implement a machine learning pipeline for **guitar string classification** using the **GuitarSet** dataset. Given a guitar audio file, the pipeline detects notes via multi-pitch estimation and classifies which string each note was played on.

If you have an idea for an improvement, found a bug, or just want to talk about the implementation, feel free to open an issue!

---

## Prerequisites

- Python 3.11
- Git LFS
- GPU recommended for YourMT3 inference (not strictly required)

---

## Setup

### 1. Install Git LFS
```bash
git lfs install
```

### 2. Clone the repository
```bash
git clone https://github.com/SimonBuechner/GuitarStringEstimation.git
cd GuitarStringEstimation
```

> If the checkpoint download fails due to Git LFS bandwidth limits, retrieve it from the original [YourMT3 repository](https://github.com/mimbres/YourMT3) and place it at `amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/checkpoints/model.ckpt`. A migration to Hugging Face is planned.

### 3. Create a Python environment

Using `venv`:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

*(Alternatively, conda works fine too.)*

### 4. Download the dataset

Download the **GuitarSet** dataset and place it at:
data/GuitarSet/

---

## Project Structure
```
GuitarStringEstimation/
├── .venv/
├── amt/                      # YourMT3 code and checkpoints
│   ├── content/
│   ├── logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/
│   │   └── checkpoints/model.ckpt
│   └── src/
│       ├── model/
│       ├── utils/            # Utilities from YourMT3
│       └── ...
├── gse/                      # Guitar string estimation code
│   └── src/
│       ├── beta_distributions.py
│       ├── YMT3_inference.py # Calls YourMT3 for note detection
│       ├── svm_full.joblib   # Pretrained SVM for string classification
│       └── ...
├── data/
│   └── GuitarSet/
│       ├── annotation/*.jams
│       ├── audio_mono-mic/
│       └── ...
├── scripts/                  # Miscellaneous experiment scripts (see below)
├── .gitignore
├── .gitattributes
├── LICENSE
└── README.md
```
---

## Usage

Run the following steps in order to reproduce the results from the paper.

### 1. Extract tracks and notes
```bash
python extract_GuitarSet.py
```
Creates a `noteData/` directory containing serialized (pickle) note data from GuitarSet.

### 2. Run YourMT3 inference
```bash
python YMT3_inference.py
```
Runs the YourMT3 model on train and test subsets, extracts predicted notes, and matches them with ground truth annotations.

> ⚠️ All subsequent steps operate only on these matched notes.

### 3. Train the SVM classifier
```bash
python train_classifier.py
```
Extracts features from the note data and trains an SVM string classifier. This step is **computationally expensive** but only needs to be run once.

> A pretrained model (`svm_full.joblib`) is already provided so you can skip this step for evaluation.

### 4. Evaluate the classifier
```bash
python evaluate_classifier.py
```
Runs string classification and evaluation. Includes options to reproduce the **permutation importance experiments** from the paper.

### 5. Evaluate inharmonicity coefficient estimation
```bash
python beta_distributions.py
```
Collects inharmonicity coefficients (β) for all strings, normalizes them to open-string equivalents, and generates plots for selected subsets.

---

## Scripts

The `scripts/` directory contains miscellaneous scripts used for additional experiments.

> ⚠️ These are not guaranteed to work out of the box. They were primarily used to evaluate other state-of-the-art approaches (e.g., Hjerrild et al., 2019) and are provided for reference only.

---

## Notes & Tips

- Feature computation can take a significant amount of time and uses multiprocessing.
- Ensure the GuitarSet directory structure matches what is shown above; mismatches will cause script failures.
- GPU is recommended for YourMT3 inference but not strictly required.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{buechner2026guitarstring,
  title     = {Bestimmung gespielter Gitarrensaiten anhand von Audiofeatures basierend auf multipler Grundfrequenzsch{\"a}tzung},
  author    = {B{\"u}chner, Simon and Bereuter, Paul A. and Sontacchi, Alois},
  year      = {2026},
}
```

---

## Based On

- [YourMT3](https://github.com/mimbres/YourMT3) — Multi-pitch music transcription transformer
- [GuitarSet](https://guitarset.weebly.com/) — Guitar recordings with detailed annotations
