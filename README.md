# Guitar String Estimation on GuitarSet via YourMT3

## Description
This project was developed at IEM Graz. It builds on **YourMT3** to implement a machine learning pipeline for **guitar string classification** using the **GuitarSet** dataset.

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

### 3. Create a Python environment (Python 3.11 recommended)

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

*(Alternatively, you can use conda if preferred.)*

### 4. Install the dataset
Download the **GuitarSet** dataset and place it in:

```
data/GuitarSet/
```

---

## Project Structure

After setup, the directory should look like:

```
GuitarStringEstimation/
├── .venv/                    
├── amt/                      
│   ├── content/              
│   ├── logs/                 
│   └── src/                  
│       ├── model/        
│       ├── utils/              # Utilities from YourMT3
│       └── ...                         
├── gse/                                    
│   └── src/                  
│       ├── beta_distributions.py        
│       ├── YMT3_inference.py   # Calls YourMT3 for note detection      
│       ├── svm_full.joblib     # Pretrained SVM for string classification
│       └── ...                 
├── data/                               
│   └── GuitarSet/                      
│       ├── annotation/*.jams   # GuitarSet annotations
│       ├── audio_mono-mic/     
│       └── ...                         
├── scripts/                    
├── .gitignore               
├── .gitattributes           
├── LICENSE                  
└── README.md     
```

---

## Usage Instructions

### 1. Extract tracks and notes
```bash
python extract_GuitarSet.py
```
Creates a `noteData/` directory containing serialized (pickle) note data from GuitarSet.

---

### 2. Run YourMT3 inference
```bash
python YMT3_inference.py
```
Runs the YourMT3 model on train and test subsets, extracts predicted notes, and matches them with ground truth annotations.

> ⚠️ All subsequent steps operate only on these matched notes.

---

### 3. Train the SVM classifier
```bash
python train_classifier.py
```
Trains an SVM on the preprocessed note data.

- A pretrained model (`svm_full.joblib`) is already provided for reproducibility.
- Feature extraction is performed via:
```bash
python calculate_features.py
```
This step is **computationally expensive** but only needs to be run once.

---

### 4. Evaluate the classifier
```bash
python evaluate_classifier.py
```
Runs classification and evaluation via CLI.

- Includes options to reproduce **feature importance experiments**
- Reuses computed features from `calculate_features.py`

---

### 5. Evaluate inharmonicity coefficient estimation
```bash
python beta_distributions.py
```
- Collects inharmonicity coefficients (β) for all strings  
- Normalizes them to open-string equivalents  
- Generates plots for selected subsets  

---

## Notes & Tips
- Feature computation can take a significant amount of time—consider caching results.
- Ensure correct dataset structure; mismatches may cause failures in scripts.
- GPU is recommended for YourMT3 inference but not strictly required.

---

## Based on
- [YourMT3](https://github.com/mimbres/YourMT3) – A modified Music Transcription Transformer model  
- [GuitarSet](https://guitarset.weebly.com/) – Guitar recordings with detailed annotations  

---

## Scripts
Contains miscellaneous scripts used for additional experiments.

> ⚠️ These are not guaranteed to work out of the box and were primarily used for evaluating other state-of-the-art approaches (e.g., Hjerrild et al., 2019).

---

## Possible Improvements
- Add example outputs or logs  
- Provide a minimal end-to-end demo script  
- Include evaluation metrics (accuracy, F1 score, etc.)  
- Add a citation / BibTeX section if linked to a paper  
- Pin dependency versions for reproducibility  