# Guitar String Estimation Sub-Repo for working YMT3 implementation

## Description

A project developed at IEM Graz during WS24/25. It uses YourMT3 as a base for an algorithm for guitar string estimation on acoustic guitars.

This repository implements a system for automatic recognition and assignment of played guitar strings in audio recordings based on the [YourMT3](https://github.com/mimbres/YourMT3) model for F0-tracking. It is currently WIP and further instructions and restructuring will follow.

## Installation

### 1. Prerequisites

**Git LFS** must be installed before cloning:

- **Mac:** `brew install git-lfs`
- **Windows:** Download from [git-lfs.com](https://git-lfs.com) or `winget install GitHub.GitLFS`

Then run once:
```bash
git lfs install
```

**Python 3.11.9** must be installed:

- **Mac:** `brew install python@3.11`
- **Windows:** Download from [python.org/downloads](https://www.python.org/downloads/release/python-3119/)

### 2. Clone the repository

Cloning without automatic LFS download (faster):

**Mac / Linux:**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sbch22/GuitarStringEstimation
cd GuitarStringEstimation
```

**Windows (PowerShell):**
```powershell
$env:GIT_LFS_SKIP_SMUDGE=1; git clone https://github.com/sbch22/GuitarStringEstimation
cd GuitarStringEstimation
```

### 3. Download checkpoints (~3.5 GB)

```bash
git lfs pull
```

This may take a few minutes depending on your connection.

### 4. Create virtual environment

**Mac / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run the demo

```bash
cd amt/src
python YMT3_demo.py
```

- Model type and config can be chosen in `main()`
- Filepath to audio file can be chosen in `main()`
- Input audio files: `amt/content/`
- Output directory: `amt/content/model_output/`

Two example audio tracks are included in the repository.

## Based on

- [YourMT3](https://github.com/mimbres/YourMT3) - A modified Music Transcription Transformer model