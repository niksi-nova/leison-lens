# LesionLens

Multi-task DR grading + lesion localization with explainability validation.

## Setup

### 1. Clone the repo
git clone https://github.com/niksi-nova/lesionlens.git
cd lesionlens

### 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Download APTOS dataset
- Get your kaggle.json from kaggle.com → Settings → API
- Place it at ~/.kaggle/kaggle.json
- chmod 600 ~/.kaggle/kaggle.json
- Accept competition rules at:
  https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/aptos

### 4. Run preprocessing (Phase 1)
python src/preprocess.py

### 5. Verify
python src/dataset.py


Clone the repo: github.com/niksi-nova/lesionlens — follow the README to set up. Download APTOS yourself using your own Kaggle account (instructions in README). Run preprocess.py once to generate the processed images locally.