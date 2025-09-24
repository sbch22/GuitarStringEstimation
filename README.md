# Guitar String Estimation on GuitarSet via YourMT3

## Description
A project developed at IEM Graz during WS24/25. It uses YourMT3 as a base for an algorithm for guitar string estimation on acoustic guitars.

This repository implements a system for automatic recognition and assignment of played guitar strings in audio recordings based on the [YourMT3](https://github.com/mimbres/YourMT3) model for F0-tracking. It is currently WIP and further instructions and restructuring will follow.

## Installation

1. Set up Git LFS:
```bash
git lfs install
```

2. Clone the repository:
```bash
git clone https://github.com/SimonBuechner/GuitarStringEstimation.git
cd GuitarStringEstimation
```

3. Install Python 3.11 in a new environment (recommended with conda or venv):
```bash
Create virtual environment: `python -m venv .venv`
Install dependencies: `pip install -r requirements.txt`

```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the dataset:
```bash
python install_dataset.py
```
Follow the instructions in the CLI. For this project, only GuitarSet dataset is needed.

6. Extract notes information the dataset:
```bash
python extract_dataset.py
```

## Dataset

This project uses the [GuitarSet Dataset](https://guitarset.weebly.com/), which contains audio recordings and annotations of acoustic guitar performances.

## Based on

- [YourMT3](https://github.com/mimbres/YourMT3) - A modified Music Transcription Transformer model
- [GuitarSet](https://guitarset.weebly.com/) - Dataset for guitar recordings with annotations

# GuitarStringEstimation Project Structure

After installation, the project directory should look like this:

```
GuitarStringEstimation/
├── .venv/                    
├── amt/                      
│   ├── content/              
│   ├── logs/                 
│   └── src/                  
│       ├── f0-tracking_accuracy/ 
│       ├── model/        
│       ├── .../                        # multiple directories from YourMT3 model        
│       ├── GuitarStringEstimator.py 
│       └── ...                         # multiple .py files referred to in Usage Instructions
├── data/                               # Dataset directory
│   ├── guitarset_yourmt3_16k/ 
│   ├── yourmt3_indexes/      
│   └── logs/                 
├── scripts/                            # Additional scripts
├── .gitignore               
├── .gitattributes           
├── LICENSE                  
└── README.md                

```


## Usage Instructions:

### Evaluate accuracy of YourMT3
To run the evaluation script of YourMT3, navigate to the appropriate directory:
```bash
cd amt/src/f0-tracking_accuracy
python f0-trackingYMT3_eval_controller.py
```

Note: These are intensive calculations best performed on a GPU-accelerated machine. For debugging, modify the `dbg` boolean toggle in the main() call in `f0-trackingYMT3_eval.py`.

### Calculate Beta-Distributions
Calculates the inharmonicity coefficient (beta) distributions:
```bash
python betaDistributions.py
```

### Perform statistical analysis on Beta-Distributions
Runs statistical tests and generates histograms:
```bash
python betaDistributions_statTest.py
```


### Run Guitar String Estimation
Calculates guitar string estimations for each note in GuitarSet:
```bash
python guitar_string_estimator.py
```

## Scripts
Found in the `scripts/` folder.
### Frequency Estimation Algorithms
Frequency estimation algorithm comparison. 
```bash
python FrequencyEstimationAccuracy_eval.py
```


### Evaluate Beta-Distributions algorithm
Evaluates the algorithm for finding Beta-Distributions on synthetic testsignals. Must be run from Project root.
```bash
python betaDistributions_eval.py
```
