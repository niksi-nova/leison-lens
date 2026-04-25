# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Main training script. Run this on Kaggle (T4 GPU) or locally.
#
# What happens each epoch:
#   1. Forward pass on training batches
#   2. Compute multi-task loss
#   3. Backprop + optimizer step
#   4. Evaluate on validation set
#   5. Save checkpoint if best QWK so far
#   6. Apply LR scheduler (ReduceLROnPlateau)
#
# Ablation:
#   Run with LAMBDA_WEIGHT = 0.3, 0.4, 0.5 and compare val QWK.
#   This becomes Table 1 (Ablation Study) in your paper.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src/ to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
                    BATCH_SIZE, NUM_EPOCHS, LAMBDA_WEIGHT, SEED, DEVICE,
                    TRAIN_RATIO, VAL_RATIO)
from model    import build_model
from dataset  import build_dataloaders
from evaluate import evaluate, print_metrics
from utils    import save_checkpoint, plot_training_curves

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting all random seeds ensures your results are reproducible —
# required for academic papers.
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)


def load_aptos_splits():

    """
    Load preprocessed splits (already created in preprocess.py)
    """
    train_csv = os.path.join(OUTPUT_DIR, "train_labels.csv")
    val_csv   = os.path.join(OUTPUT_DIR, "val_labels.csv")
    test_csv  = os.path.join(OUTPUT_DIR, "test_labels.csv")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    print(f"[split] train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one full training epoch.

    Returns:
        avg_loss, avg_grade_loss, avg_lesion_loss
    """
    model.train()   # enables dropout + batch norm training mode
    total_loss = grade_total = lesion_total = 0.0

    for batch_idx, (images, grade_labels, lesion_labels) in enumerate(loader):
        images        = images.to(device, non_blocking=True)
        grade_labels  = grade_labels.to(device, non_blocking=True)
        lesion_labels = lesion_labels.to(device, non_blocking=True)

        optimizer.zero_grad()   # clear gradients from previous batch

        # ── Forward pass ──────────────────────────────────────────────────────
        grade_logits, lesion_logits = model(images)

        # ── Compute loss ──────────────────────────────────────────────────────
        loss, g_loss, l_loss = criterion(
            grade_logits, lesion_logits,
            grade_labels, lesion_labels
        )

        # ── Backward pass ─────────────────────────────────────────────────────
        loss.backward()   # compute gradients via autograd

        # Gradient clipping: prevents exploding gradients
        # (important for fine-tuning pretrained models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()   # update weights

        total_loss  += loss.item()
        grade_total += g_loss.item()
        lesion_total+= l_loss.item()

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss={loss.item():.4f} "
                  f"(Grade={g_loss.item():.4f}, Lesion={l_loss.item():.4f})")

    n = len(loader)
    return total_loss/n, grade_total/n, lesion_total/n


def main(lambda_weight: float = LAMBDA_WEIGHT):
    """
    Full training pipeline.
    Pass lambda_weight as argument for ablation experiments.
    """
    print("=" * 60)
    print(f"  DR Multi-Task Training | λ={lambda_weight} | Device={DEVICE}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_aptos_splits()

    # Image directories (preprocessed images from Phase 2)
    #img_dir = os.path.join(OUTPUT_DIR, "aptos")  # adjust if your structure differs
    from config import TRAIN_OUTPUT_DIR, VAL_OUTPUT_DIR, TEST_OUTPUT_DIR

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        batch_size    = BATCH_SIZE,
        img_dir_train = str(TRAIN_OUTPUT_DIR),
        img_dir_val   = str(VAL_OUTPUT_DIR),
        img_dir_test  = str(TEST_OUTPUT_DIR),
    )
    

    # ── Model ─────────────────────────────────────────────────────────────────
    model, criterion, optimizer = build_model(
        lambda_weight = lambda_weight,
        device        = DEVICE
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    # ReduceLROnPlateau: halves LR if val QWK doesn't improve for 3 epochs.
    # This is adaptive — no need to manually tune LR decay schedule.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode     = 'max',
    factor   = 0.5,
    patience = 3
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_qwk     = 0.0
    train_losses = []
    val_losses   = []
    val_kappas   = []

    checkpoint_name = f"best_model_lambda{lambda_weight}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{NUM_EPOCHS} ─────────────────────────────────")

        # Train
        train_loss, g_loss, l_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )
        print(f"\n  [Train] Loss={train_loss:.4f} | "
              f"Grade={g_loss:.4f} | Lesion={l_loss:.4f}")

        # Evaluate on validation set
        metrics = evaluate(model, val_loader, criterion, DEVICE)
        print_metrics(metrics, epoch=epoch)

        val_qwk  = metrics["qwk"]
        val_loss = metrics["val_loss"]

        # Log for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_kappas.append(val_qwk)

        # Step scheduler based on val QWK
        scheduler.step(val_qwk)

        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            save_checkpoint(model, optimizer, epoch, val_qwk,
                            CHECKPOINT_DIR, checkpoint_name)
            print(f"  ★ New best QWK: {best_qwk:.4f}")

    # ── Final evaluation on test set ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 60)

    # Load best model
    from utils import load_checkpoint
    load_checkpoint(model, None, CHECKPOINT_DIR, checkpoint_name, DEVICE)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    print_metrics(test_metrics)
    print(f"\nFinal Test QWK: {test_metrics['qwk']:.4f}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, val_kappas, LOG_DIR)

    print("\nTraining complete. Checkpoints saved to:", CHECKPOINT_DIR)
    print("Plots saved to:", LOG_DIR)

    return best_qwk


# ── Ablation runner ───────────────────────────────────────────────────────────
# Uncomment the block below to run the full lambda ablation study.
# Each run saves a separate checkpoint with the lambda in the filename.
# Results go into Table 1 of your paper.

# if __name__ == "__main__":
#     results = {}
#     for lam in [0.3, 0.4, 0.5]:
#         print(f"\n\n{'='*60}")
#         print(f"  ABLATION: lambda = {lam}")
#         print(f"{'='*60}\n")
#         qwk = main(lambda_weight=lam)
#         results[lam] = qwk
#
#     print("\n── Ablation Results ──────────────────────────")
#     for lam, qwk in results.items():
#         print(f"  λ={lam} → Test QWK = {qwk:.4f}")

if __name__ == "__main__":
    main()