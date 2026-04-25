# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Utility functions:
#   1. compute_class_weights  — handles APTOS class imbalance
#   2. save_checkpoint        — saves model + optimizer state
#   3. load_checkpoint        — resumes training or loads for inference
#   4. plot_training_curves   — saves loss/accuracy graphs to logs/
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights so the DataLoader
    samples rare classes (e.g. Severe DR, Grade 4) more often.

    Formula: weight[c] = total_samples / (num_classes * count[c])
    This ensures each class contributes equally to the gradient updates.

    Args:
        labels      : list of integer class labels for the training set
        num_classes : total number of classes (5 for DR grades)

    Returns:
        torch.Tensor of shape [N] — one weight per sample (for WeightedRandomSampler)
    """
    count = Counter(labels)
    total = len(labels)

    # Weight per CLASS — rarer class → higher weight
    class_weight = {
        c: total / (num_classes * count[c])
        for c in range(num_classes)
    }

    # Weight per SAMPLE — assign each sample its class's weight
    sample_weights = [class_weight[lbl] for lbl in labels]

    print("[utils] Class distribution:", dict(count))
    print("[utils] Class weights:", {c: f"{w:.3f}" for c, w in class_weight.items()})

    return torch.tensor(sample_weights, dtype=torch.float)


def save_checkpoint(model, optimizer, epoch: int, val_kappa: float,
                    checkpoint_dir: str, filename: str = "best_model.pth"):
    """
    Saves model weights, optimizer state, epoch number, and best kappa score.
    Allows training to be resumed exactly where it stopped.

    Args:
        model          : the DRMultiTaskModel instance
        optimizer      : the optimizer (AdamW)
        epoch          : current epoch number
        val_kappa      : quadratic weighted kappa on validation set
        checkpoint_dir : directory to save the .pth file
        filename       : name of the checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)

    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_kappa":  val_kappa,
    }, path)

    print(f"[checkpoint] Saved → {path}  (epoch={epoch}, kappa={val_kappa:.4f})")


def load_checkpoint(model, optimizer, checkpoint_dir: str,
                    filename: str = "best_model.pth", device: str = "cpu"):
    """
    Loads a saved checkpoint back into model and optimizer.
    Call this to resume training or to run inference.

    Returns:
        (start_epoch, best_kappa) — so the training loop can continue correctly
    """
    path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(path):
        print(f"[checkpoint] No checkpoint found at {path}. Starting fresh.")
        return 0, 0.0

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"[checkpoint] Loaded ← {path}  (epoch={ckpt['epoch']}, kappa={ckpt['val_kappa']:.4f})")
    return ckpt["epoch"], ckpt["val_kappa"]


def plot_training_curves(train_losses: list, val_losses: list,
                         val_kappas: list, log_dir: str):
    """
    Saves two plots to logs/:
      1. training_loss.png  — train vs val loss per epoch
      2. val_kappa.png      — quadratic weighted kappa per epoch

    These plots go directly into the Results section of your paper.
    """
    os.makedirs(log_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # ── Plot 1: Loss curves ──
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses,   label="Val Loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_loss.png"), dpi=150)
    plt.close()
    print(f"[plot] Saved training_loss.png → {log_dir}")

    # ── Plot 2: Kappa curve ──
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_kappas, label="Val Kappa (QWK)", marker="^", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Quadratic Weighted Kappa")
    plt.title("Validation Kappa per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "val_kappa.png"), dpi=150)
    plt.close()
    print(f"[plot] Saved val_kappa.png → {log_dir}")