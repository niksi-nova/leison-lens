# src/preprocess.py

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm          # shows a progress bar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.config import BASE_DIR
import sys
sys.path.append(os.path.expanduser("~/el4"))
from src.config import *


# ────────────────────────────────────────────────────────────────
# FUNCTION 1: Ben Graham Preprocessing
# ────────────────────────────────────────────────────────────────
# WHY: Fundus camera images have inconsistent brightness and color
# cast depending on the camera model and patient. This technique,
# invented by the 2015 Kaggle DR competition winner, removes the
# global illumination and reveals local lesion structure.
#
# HOW: It subtracts a heavily blurred version of the image from
# the original, then adds 128 to keep pixel values in [0,255].
# The result is a "local contrast" image where vessels and lesions
# pop out regardless of original lighting.
# ────────────────────────────────────────────────────────────────
def ben_graham_preprocess(img: np.ndarray, sigma: int = BEN_SIGMA) -> np.ndarray:
    """
    Apply Ben Graham preprocessing to remove illumination bias.
    
    Args:
        img:   BGR image as numpy array (from cv2.imread)
        sigma: Gaussian blur sigma. Larger = more aggressive removal.
    
    Returns:
        Preprocessed BGR image as numpy array.
    """
    # Create a heavily blurred version of the image
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Subtract blurred from original, scale by 4, add 128 as offset
    # This is: result = 4*img - 4*blurred + 128
    result = cv2.addWeighted(img, 4, blurred, -4, 128)
    
    return result


# ────────────────────────────────────────────────────────────────
# FUNCTION 2: CLAHE on Green Channel
# ────────────────────────────────────────────────────────────────
# WHY: DR lesions (microaneurysms, haemorrhages, exudates) are most
# visible in the green channel of RGB fundus images. CLAHE (Contrast
# Limited Adaptive Histogram Equalization) boosts local contrast
# without over-amplifying noise — better than standard histogram eq.
#
# HOW: We extract the green channel, apply CLAHE to it, then put
# it back. The red and blue channels are left unchanged.
# ────────────────────────────────────────────────────────────────
def apply_clahe_green(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the green channel of a BGR image.
    
    Args:
        img: BGR image as numpy array
    
    Returns:
        BGR image with CLAHE applied to green channel
    """
    # Split into Blue, Green, Red channels
    b, g, r = cv2.split(img)
    
    # Create CLAHE object
    # clipLimit: contrast amplification is clipped at this value
    # tileGridSize: image divided into this many tiles; each gets its own histogram
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_GRID
    )
    
    # Apply only to green channel
    g_enhanced = clahe.apply(g)
    
    # Merge back: same blue and red, enhanced green
    result = cv2.merge([b, g_enhanced, r])
    
    return result


# ────────────────────────────────────────────────────────────────
# FUNCTION 3: Circular Crop
# ────────────────────────────────────────────────────────────────
# WHY: Fundus images have a circular field-of-view surrounded by
# a black border. That black ring is useless and slightly confuses
# models. We crop to a tight square around the fundus circle.
#
# HOW: Find the largest contour in a thresholded image (the fundus
# circle), get its bounding box, crop to that box.
# ────────────────────────────────────────────────────────────────
def crop_fundus_circle(img: np.ndarray) -> np.ndarray:
    """
    Crop the image to the fundus circle, removing black borders.
    
    Args:
        img: BGR image
    
    Returns:
        Cropped BGR image (may not be square yet — resize comes later)
    """
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold: any pixel brighter than 15 is considered fundus
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours of the bright regions
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # If no contours found, return original (safety fallback)
    if not contours:
        return img
    
    # Get the largest contour (the fundus circle)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add a small padding of 5 pixels so we don't cut the edge
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2*pad)
    h = min(img.shape[0] - y, h + 2*pad)
    
    return img[y:y+h, x:x+w]


# ────────────────────────────────────────────────────────────────
# FUNCTION 4: Full Pipeline for One Image
# ────────────────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single image.
    
    Steps:
        1. Read image
        2. Crop fundus circle
        3. Resize to IMG_SIZE × IMG_SIZE
        4. Ben Graham illumination removal
        5. CLAHE on green channel
    
    Args:
        img_path: Full path to the .png image
    
    Returns:
        Preprocessed image as numpy array (BGR, uint8, 512×512×3)
        Returns None if image cannot be read.
    """
    # Step 1: Read
    img = cv2.imread(img_path)
    if img is None:
        print(f"WARNING: Could not read {img_path}")
        return None
    
    # Step 2: Crop black border
    img = crop_fundus_circle(img)
    
    # Step 3: Resize to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Step 4: Ben Graham
    img = ben_graham_preprocess(img)
    
    # Step 5: CLAHE on green channel
    img = apply_clahe_green(img)
    
    return img


# ────────────────────────────────────────────────────────────────
# FUNCTION 5: Split CSV into Train/Val/Test
# ────────────────────────────────────────────────────────────────
def split_dataset(csv_path: str) -> tuple:
    """
    Load train.csv and split into train/val/test dataframes.
    Uses stratified splitting to maintain class distribution in each split.
    
    Args:
        csv_path: Path to train.csv
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    
    print(f"Total images: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")
    
    # First split: separate test set
    # stratify=df['diagnosis'] ensures each split has proportional class counts
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df['diagnosis']
    )
    
    # Second split: separate val from train
    # val_size relative to train_val portion
    relative_val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_SEED,
        stratify=train_val_df['diagnosis']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    
    return train_df, val_df, test_df


# ────────────────────────────────────────────────────────────────
# FUNCTION 6: Process and Save All Images
# ────────────────────────────────────────────────────────────────
def process_and_save_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Process all images in a dataframe split and save them to disk.
    Also saves a CSV with the split's labels.
    
    Args:
        df:         Dataframe with id_code and diagnosis columns
        split_name: One of 'train', 'val', 'test'
    
    Returns:
        Dataframe updated with 'processed_path' column
    """
    out_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    
    processed_paths = []
    failed = []
    
    print(f"\nProcessing {split_name} split ({len(df)} images)...")
    
    # tqdm wraps the iteration to show a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        img_id = row['id_code']
        src_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")

        if not os.path.exists(src_path):
            failed.append(img_id)
            processed_paths.append(None)
            continue
        dst_path = os.path.join(out_dir, f"{img_id}.png")
        
        # Skip if already processed (lets you resume if interrupted)
        if os.path.exists(dst_path):
            processed_paths.append(dst_path)
            continue
        
        # Preprocess
        processed = preprocess_image(src_path)
        
        if processed is not None:
            cv2.imwrite(dst_path, processed)
            processed_paths.append(dst_path)
        else:
            print(f"FAILED: {src_path}")   # 👈 ADD THIS
            failed.append(img_id)
            processed_paths.append(None)
    
    if failed:
        print(f"  WARNING: {len(failed)} images failed: {failed[:5]}...")
    
    # Add path column and save CSV
    df = df.copy()
    df['processed_path'] = processed_paths
    csv_out = os.path.join(OUTPUT_DIR, f"{split_name}_labels.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved labels to {csv_out}")
    
    return df


# ────────────────────────────────────────────────────────────────
# FUNCTION 7: Visualize Before/After
# ────────────────────────────────────────────────────────────────
def visualize_preprocessing(n_samples: int = 4):
    """
    Show before/after comparison for n random images.
    Saves the figure to outputs/preprocessing_check.png
    """
    df = pd.read_csv(TRAIN_CSV)
    sample = df.sample(n=n_samples, random_state=RANDOM_SEED)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 4))
    fig.suptitle("Preprocessing: Before vs After", fontsize=16, y=1.01)
    
    for i, (_, row) in enumerate(sample.iterrows()):
        src_path = os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png")
        label = CLASS_NAMES[row['diagnosis']]
        
        # Original (convert BGR→RGB for matplotlib)
        original = cv2.imread(src_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Processed
        processed = preprocess_image(src_path)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f"Original — {label}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(processed_rgb)
        axes[i, 1].set_title(f"Preprocessed — {label}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "preprocessing_check.png"

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    print(f"\nSaved visualization to {out_path}")


# ────────────────────────────────────────────────────────────────
# FUNCTION 8: Compute Class Weights
# ────────────────────────────────────────────────────────────────
# WHY: APTOS has severe class imbalance — there are ~1800 No DR
# images but only ~193 Severe DR images. Without correction, the
# model will just learn to predict No DR and ignore rare classes.
# We compute weights inversely proportional to class frequency
# so the loss penalizes mistakes on rare classes more.
# ────────────────────────────────────────────────────────────────
def compute_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute inverse-frequency class weights for weighted loss.
    
    Returns:
        numpy array of shape (5,) with weight per class
    """
    counts = train_df['diagnosis'].value_counts().sort_index().values
    total  = counts.sum()
    
    # weight = total / (num_classes × count_of_class)
    weights = total / (NUM_CLASSES * counts)
    weights = weights / weights.sum() * NUM_CLASSES   # normalize to sum to 5
    
    print("\nClass weights (higher = model penalized more for errors):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"  Class {i} ({name:20s}): {w:.3f}  [count: {counts[i]}]")
    
    return weights.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# MAIN — Run the full pipeline
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step A: Visualize a few images before/after (sanity check first)
    print("\n[1/4] Generating before/after visualization...")
    visualize_preprocessing(n_samples=4)
    
    # Step B: Split the dataset
    print("\n[2/4] Splitting dataset...")
    train_df, val_df, test_df = split_dataset(TRAIN_CSV)
    
    # Step C: Process and save all splits
    print("\n[3/4] Processing and saving images...")
    train_df = process_and_save_split(train_df, 'train')
    val_df   = process_and_save_split(val_df,   'val')
    test_df  = process_and_save_split(test_df,  'test')
    
    # Step D: Compute class weights (saved for Phase 2 use)
    print("\n[4/4] Computing class weights...")
    weights = compute_class_weights(train_df)
    np.save(
        os.path.join(OUTPUT_DIR, "class_weights.npy"),
        weights
    )
    print(f"\nWeights saved to outputs/preprocessed/class_weights.npy")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"Preprocessed images saved to: {OUTPUT_DIR}")
    print("=" * 60)