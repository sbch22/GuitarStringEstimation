# Guitar String Estimation Sub-Repo for working YMT3 implementation

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

3. Install Python 3.11.9 in a new environment (recommended with conda):
```bash
# With pyenv + venv:
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# With conda:
conda create -n guitarstring python=3.11.9
conda activate guitarstring
```
4. Install Dependencies
```bash
Install dependencies: `pip install -r requirements.txt`

```

5. Extract a MIDI from a local audio file in ../content/
```bash
python3 YMT3_demo.py
```
- Model type (+config) can be chosen in main()
- Filepath to audio file can be chosen in main()
- Output directory: ../content/model_output/

Note that two tracks are added via the git-repo. 


## Based on
- [YourMT3](https://github.com/mimbres/YourMT3) - A modified Music Transcription Transformer model
