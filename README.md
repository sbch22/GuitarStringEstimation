# Guitar String Estimation on GuitarSet via YMT3+

## Description
A project developed at IEM Graz during WS24/25. It uses YMT3 as a base for an algorithm for guitar string estimation on acoustic guitars.

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
conda create -n guitar_string_estimation python=3.11
conda activate guitar_string_estimation
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the dataset:
```bash
python install_dataset.py
```
Follow the instructions in the CLI. For this project, only the GuitarSet dataset is needed, no checkpoints.

6. Extract note information the dataset:
```bash
python extract_dataset.py
```

## Dataset

This project uses the [GuitarSet Dataset](https://guitarset.weebly.com/), which contains audio recordings and annotations of acoustic guitar performances.

## Based on

- [YourMT3](https://github.com/mimbres/YourMT3) - A modified Music Transcription Transformer model
- [GuitarSet](https://guitarset.weebly.com/) - Dataset for guitar recordings with annotations

## Project Structure


## Evaluate accuracy of YMT3
To run the evaluation script of YMT3, please run controller for said script. The different models are calculated on the full dataset (could be specified in script) and compares individual models.

Note that these are some intense calculations and should be performed on GPU-accelerated machine.
```bash
python f0-trackingYMT3_eval_controller.py
```



## Calculate Beta-Distributions
Calculates the inharmonicity coefficient (beta) distributions over the debleeded hex signals and writes them temporarily as betas.json

Note that these are some intense calculations and should be performed on GPU-accelerated machine.
```bash
python betaDistributions.py
```

## Perform statistical analysis on Beta-Distributions
Performs simple statistical tests on Beta-Distributions and plots Histograms from 'betas.json'. This is not neccessary for the workflow of the algorithm but can give further insight.

```bash
python betaDistributions_stat-test.py
```

## Evaluate Beta-Distributions algorithm
Evaluates the algorithm for finding the Beta-Distributions from GuitarSet.
```bash
python betaDistributions_eval.py
```


# Run Guitar String Estimation
Calculates guitar string estimations for each note in GuitarSet (mono-pickup), collects the results and gives feedback on accuracy.
```bash
python GuitarStringEstimator.py
```