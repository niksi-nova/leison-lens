# src/config.py
from pathlib import Path
import torch

# ── Base project directory ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data" / "aptos"

TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR  = DATA_DIR / "test_images"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# ── Output / Logging / Checkpoints ──────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "outputs" / "preprocessed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

TRAIN_OUTPUT_DIR = OUTPUT_DIR / "train"
VAL_OUTPUT_DIR   = OUTPUT_DIR / "val"
TEST_OUTPUT_DIR  = OUTPUT_DIR / "test"

# ── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE = 512
NUM_CHANNELS = 3

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BEN_SIGMA  = 10

# ── Model settings ───────────────────────────────────────────────────────────
BACKBONE = "efficientnet_b4"

NUM_CLASSES = 5          # DR grade classes
NUM_LESION_CLASSES = 4   # MA, HE, EX, SE

DROPOUT_RATE = 0.4

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# ── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE    = 12    
NUM_EPOCHS    = 30   
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5

# ── Multi-task Loss Weight ──────────────────────────────────────────────────
LAMBDA_WEIGHT = 0.4

# ── Dataset Split Ratios ────────────────────────────────────────────────────
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10

# ✅ Aliases (to match train.py expectations)
TRAIN_RATIO = TRAIN_SPLIT
VAL_RATIO   = VAL_SPLIT
TEST_RATIO  = TEST_SPLIT

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
SEED = RANDOM_SEED

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"[config] Using device: {DEVICE}")

# ── Auto-create directories ────────────────────────────────────────────────
for path in [
    OUTPUT_DIR,
    TRAIN_OUTPUT_DIR,
    VAL_OUTPUT_DIR,
    TEST_OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# ── Debug ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[config] Using device: {DEVICE}")
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("TRAIN_CSV exists:", TRAIN_CSV.exists())
    print("TRAIN_IMG_DIR exists:", TRAIN_IMG_DIR.exists())