# src/dataset.py

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.expanduser("~/el4"))
from src.config import *


# ────────────────────────────────────────────────────────────────
# Augmentation Transforms
# ────────────────────────────────────────────────────────────────
# WHY AUGMENTATION: Our training set is 2929 images. That's small
# for a deep learning model. Augmentation artificially expands
# diversity by applying random transforms — flips, rotations, etc.
# — while preserving the disease label. A flipped DR image is still
# a DR image.
#
# We use albumentations (faster than torchvision transforms for
# image augmentation, especially with OpenCV backend).
#
# Training gets heavy augmentation. Val/Test get only normalization.
# ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> A.Compose:
    """
    Return albumentations transform pipeline for a given split.
    
    Args:
        split: 'train', 'val', or 'test'
    """
    
    # ImageNet mean and std — used because our backbone was pretrained on ImageNet
    # All ImageNet-pretrained models expect inputs normalized this way
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    if split == 'train':
        return A.Compose([
            # Random horizontal flip — 50% chance
            A.HorizontalFlip(p=0.5),
            
            # Random vertical flip — fundus images are symmetric
            A.VerticalFlip(p=0.5),
            
            # Random rotation up to 360 degrees — retina has no orientation
            A.Rotate(limit=360, p=0.8),
            
            # Slight random brightness/contrast shift
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Random zoom in/out
            A.RandomScale(scale_limit=0.1, p=0.3),
            
            # Ensure output is still IMG_SIZE after zoom
            A.Resize(IMG_SIZE, IMG_SIZE),
            
            # Normalize to ImageNet stats, then convert to PyTorch tensor
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    else:   # val or test — no random transforms
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


# ────────────────────────────────────────────────────────────────
# Dataset Class
# ────────────────────────────────────────────────────────────────
class APTOSDataset(Dataset):
    """
    PyTorch Dataset for the APTOS fundus image dataset.
    
    Loads preprocessed images and applies transforms.
    Returns (image_tensor, label) pairs.
    """
    
    def __init__(self, csv_path: str, split: str = 'train'):
        """
        Args:
            csv_path: Path to the split's label CSV 
                      (e.g. outputs/preprocessed/train_labels.csv)
            split:    'train', 'val', or 'test'
        """
        self.df        = pd.read_csv(csv_path)
        self.split     = split
        self.transform = get_transforms(split)
        
        # Drop any rows where preprocessing failed (processed_path is None)
        self.df = self.df.dropna(subset=['processed_path'])
        self.df = self.df.reset_index(drop=True)
        
        print(f"Loaded {split} dataset: {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row['diagnosis'])
        
        # Load preprocessed image (BGR from OpenCV)
        img = cv2.imread(row['processed_path'])
        
        # Convert BGR → RGB (albumentations and PyTorch expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms — albumentations expects numpy HWC uint8
        transformed = self.transform(image=img)
        img_tensor  = transformed['image']    # now a CHW float tensor
        
        return img_tensor, torch.tensor(label, dtype=torch.long)
    
    def get_labels(self):
        """Return all labels as a list — needed for WeightedRandomSampler."""
        return self.df['diagnosis'].tolist()


# ────────────────────────────────────────────────────────────────
# DataLoader Factory
# ────────────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 16) -> dict:
    """
    Create train/val/test DataLoaders.
    
    Train uses WeightedRandomSampler to handle class imbalance.
    Val and Test use regular sequential sampling.
    
    Args:
        batch_size: Images per batch. 16 is safe for M5 Pro RAM.
    
    Returns:
        Dict with keys 'train', 'val', 'test' → DataLoader objects
    """
    splits = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(OUTPUT_DIR, f"{split}_labels.csv")
        dataset  = APTOSDataset(csv_path, split=split)
        
        if split == 'train':
            # Load precomputed class weights
            class_weights = np.load(
                os.path.join(OUTPUT_DIR, "class_weights.npy")
            )
            
            # Assign a weight to each sample based on its class
            sample_weights = [
                class_weights[label] for label in dataset.get_labels()
            ]
            sample_weights = torch.tensor(sample_weights, dtype=torch.float)
            
            # WeightedRandomSampler: over-samples rare classes, 
            # under-samples common ones, so each batch is balanced
            sampler = WeightedRandomSampler(
                weights     = sample_weights,
                num_samples = len(sample_weights),
                replacement = True
            )
            
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,      # mutually exclusive with shuffle
                num_workers = 0,            # 0 is safest on Mac M-series
                pin_memory  = False
            )
        
        else:
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = 0,
                pin_memory  = False
            )
        
        splits[split] = loader
    
    return splits


# ────────────────────────────────────────────────────────────────
# Quick sanity test — run this file directly to verify
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DataLoader setup...")
    
    loaders = get_dataloaders(batch_size=4)
    
    # Grab one batch from train
    imgs, labels = next(iter(loaders['train']))
    
    print(f"\nBatch image tensor shape: {imgs.shape}")   
    # Expected: torch.Size([4, 3, 512, 512])
    print(f"Batch labels: {labels}")                    
    # Expected: tensor of 4 values in [0,4]
    print(f"Image dtype:  {imgs.dtype}")                
    # Expected: torch.float32
    print(f"Image range:  [{imgs.min():.2f}, {imgs.max():.2f}]")
    # Expected: approximately [-2.1, 2.6] after ImageNet normalization
    
    print("\nDataLoader test passed!")