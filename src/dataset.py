# dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset class for APTOS 2019.
#
# What this file does:
#   1. Reads the CSV (image_id, diagnosis columns)
#   2. Loads preprocessed images from output/ directory
#   3. Applies albumentations augmentations (training only)
#   4. Returns (image_tensor, grade_label, lesion_label) per sample
#
# Lesion labels: Since APTOS doesn't have pixel-level annotations,
# we create PROXY lesion labels from the grade:
#   Grade 0 → [0,0,0,0]  (no lesions)
#   Grade 1 → [1,0,0,0]  (MA only)
#   Grade 2 → [1,1,0,0]  (MA + HE)
#   Grade 3 → [1,1,1,0]  (MA + HE + EX)
#   Grade 4 → [1,1,1,1]  (all lesions)
# These are used to pre-train the lesion head — IDRiD provides real masks later.
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMG_SIZE, OUTPUT_DIR, NUM_LESION_CLASSES


# Proxy lesion label mapping from DR grade
# Each position = [Microaneurysms, Hemorrhages, Exudates, Soft Exudates]
GRADE_TO_LESION = {
    0: [0, 0, 0, 0],
    1: [1, 0, 0, 0],
    2: [1, 1, 0, 0],
    3: [1, 1, 1, 0],
    4: [1, 1, 1, 1],
}


def get_transforms(mode: str) -> A.Compose:
    """
    Returns albumentations augmentation pipeline.

    Training augmentations are aggressive because retinal images have
    natural rotational and flip symmetry — the retina looks the same
    from any angle. This prevents overfitting significantly.

    Validation/test: only normalize and convert to tensor — no randomness.
    """
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),           # retina is symmetric
            A.VerticalFlip(p=0.5),             # valid augmentation for fundus
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=30,
                p=0.5
            ),
            # Colour jitter — simulates different camera/lighting conditions
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            # Coarse dropout = randomly black out patches (forces model to
            # not rely on single regions, improves generalisation)
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, fill_value=0, p=0.3
            ),
            # ImageNet normalisation — required because we use ImageNet pretrained weights
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),   # converts HxWxC numpy → CxHxW torch tensor
        ])
    else:
        # Val/test: no augmentation, just normalise
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Blindness Detection.

    Args:
        dataframe : pandas DataFrame with columns ['image_id', 'diagnosis']
        mode      : 'train', 'val', or 'test'
        img_dir   : directory containing preprocessed .png images
    """

    def __init__(self, dataframe: pd.DataFrame, mode: str = "train",
             img_dir: str = None):
    
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode

        # ✅ DEFINE img_dir BEFORE USING IT
        self.img_dir = img_dir or os.path.join(OUTPUT_DIR, "aptos", mode)

        self.transform = get_transforms(mode)

        # ✅ FILTER VALID IMAGES (same logic, just in correct order)
        valid_rows = []
        for _, row in self.df.iterrows():
            img_id = row["id_code"] if "id_code" in row else row["image_id"]
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[dataset] Filtered valid samples: {len(self.df)}")

        print(f"[dataset] {mode.upper()} set: {len(self.df)} samples | "
            f"img_dir={self.img_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_id = row["id_code"] if "id_code" in row else row["image_id"]
        grade    = int(row["diagnosis"])

        # ── Load preprocessed image ───────────────────────────────────────────
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img      = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # OpenCV loads as BGR — convert to RGB for albumentations/PyTorch
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Augmentations ─────────────────────────────────────────────────────
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]   # shape: [3, 512, 512]

        # ── Labels ────────────────────────────────────────────────────────────
        grade_label  = torch.tensor(grade, dtype=torch.long)
        # Proxy lesion label from grade (multi-label binary vector)
        lesion_label = torch.tensor(
            GRADE_TO_LESION[grade], dtype=torch.float32
        )

        return img_tensor, grade_label, lesion_label


def build_dataloaders(train_df, val_df, test_df, batch_size: int,
                      img_dir_train: str, img_dir_val: str, img_dir_test: str):
    """
    Builds train/val/test DataLoaders with weighted sampling for class balance.

    Weighted Random Sampling:
    - Samples rare classes more frequently during training
    - Does NOT change the dataset — only changes which samples are drawn each epoch
    - This is better than oversampling (duplicating images) as it uses all data

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from utils import compute_class_weights
    from config import NUM_CLASSES

    train_set = APTOSDataset(train_df, mode="train", img_dir=img_dir_train)
    val_set   = APTOSDataset(val_df,   mode="val",   img_dir=img_dir_val)
    test_set  = APTOSDataset(test_df,  mode="test",  img_dir=img_dir_test)

    # ── Weighted sampler for training only ────────────────────────────────────
    train_labels    = train_df["diagnosis"].tolist()
    sample_weights  = compute_class_weights(train_labels, NUM_CLASSES)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True   # allows over-sampling of rare classes
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        sampler     = sampler,      # replaces shuffle=True
        num_workers = 2,
        pin_memory = (DEVICE == "cuda")        # speeds up CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory = (DEVICE == "cuda")
    )

    test_loader = DataLoader(
        test_set,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory = (DEVICE == "cuda")
    )

    print(f"[dataloader] train={len(train_loader)} batches | "
          f"val={len(val_loader)} | test={len(test_loader)}")

    return train_loader, val_loader, test_loader