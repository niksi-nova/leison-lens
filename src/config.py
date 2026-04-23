# src/config.py
from pathlib import Path

# ── Base project directory ─────────────────────────────
# Automatically points to the project root:
# /Users/anika/Documents/PROJECTS/el4
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────
DATA_DIR = BASE_DIR / "data" / "aptos"

TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR = DATA_DIR / "test_images"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# ── Output paths ───────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "outputs" / "preprocessed"

TRAIN_OUTPUT_DIR = OUTPUT_DIR / "train"
VAL_OUTPUT_DIR = OUTPUT_DIR / "val"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test"

# ── Image settings ─────────────────────────────────────
IMG_SIZE = 512          # Resize images to 512x512
CLAHE_CLIP = 2.0        # Contrast limit for CLAHE
CLAHE_GRID = (8, 8)     # Tile grid size for CLAHE
BEN_SIGMA = 10          # Gaussian blur sigma for Ben Graham preprocessing

# ── Dataset split ──────────────────────────────────────
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# ── Class info ─────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = [
    "No DR",              # 0
    "Mild",               # 1
    "Moderate",           # 2
    "Severe",             # 3
    "Proliferative DR"    # 4
]

# ── Create output folders automatically ───────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Debug print (optional) ─────────────────────────────
if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("TRAIN_CSV exists:", TRAIN_CSV.exists())
    print("TRAIN_IMG_DIR exists:", TRAIN_IMG_DIR.exists())