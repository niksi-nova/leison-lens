# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics for DR grading.
#
# Primary metric: Quadratic Weighted Kappa (QWK)
#   - Official APTOS competition metric
#   - Penalises large disagreements more than small ones
#   - Score of 0 = random; 1.0 = perfect; > 0.8 = clinically acceptable
#
# Secondary metrics: per-class accuracy, confusion matrix, lesion AUROC
# ─────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix
from config import DEVICE


@torch.no_grad()   # disables gradient computation — saves memory during eval
def evaluate(model, dataloader, criterion, device: str = DEVICE):
    """
    Runs one full pass over the dataloader and computes:
      - Average total loss
      - Quadratic Weighted Kappa (primary metric)
      - Per-class accuracy
      - Lesion AUROC (multi-label)
      - Confusion matrix

    Args:
        model      : DRMultiTaskModel
        dataloader : val or test DataLoader
        criterion  : MultiTaskLoss
        device     : 'cuda' or 'cpu'

    Returns:
        dict of metrics
    """
    model.eval()   # disables dropout, batchnorm uses running stats

    all_grade_preds   = []
    all_grade_labels  = []
    all_lesion_probs  = []   # sigmoid probabilities for AUROC
    all_lesion_labels = []
    total_loss = 0.0

    for batch_idx, (images, grade_labels, lesion_labels) in enumerate(dataloader):
        images        = images.to(device)
        grade_labels  = grade_labels.to(device)
        lesion_labels = lesion_labels.to(device)

        # Forward pass
        grade_logits, lesion_logits = model(images)

        # Compute loss
        loss, _, _ = criterion(grade_logits, lesion_logits,
                                grade_labels, lesion_labels)
        total_loss += loss.item()

        # Convert grade logits → predicted class (argmax)
        grade_preds = torch.argmax(grade_logits, dim=1)

        # Convert lesion logits → probabilities via sigmoid (for AUROC)
        lesion_probs = torch.sigmoid(lesion_logits)

        # Collect predictions for metric computation
        all_grade_preds.extend(grade_preds.cpu().numpy())
        all_grade_labels.extend(grade_labels.cpu().numpy())
        all_lesion_probs.extend(lesion_probs.cpu().numpy())
        all_lesion_labels.extend(lesion_labels.cpu().numpy())

    # ── Quadratic Weighted Kappa ───────────────────────────────────────────────
    # weights='quadratic' means a prediction of 4 when truth is 0 is penalised
    # 4x more than a prediction of 1 when truth is 0.
    qwk = cohen_kappa_score(
        all_grade_labels,
        all_grade_preds,
        weights='quadratic'
    )

    # ── Per-class accuracy ─────────────────────────────────────────────────────
    labels_arr = np.array(all_grade_labels)
    preds_arr  = np.array(all_grade_preds)
    per_class_acc = {}
    for c in range(5):
        mask = labels_arr == c
        if mask.sum() > 0:
            per_class_acc[f"grade_{c}"] = (preds_arr[mask] == c).mean()

    # ── Lesion AUROC ──────────────────────────────────────────────────────────
    # AUROC per lesion type — measures ranking quality (not thresholded accuracy)
    lesion_labels_arr = np.array(all_lesion_labels)
    lesion_probs_arr  = np.array(all_lesion_probs)
    lesion_names      = ["MA", "HE", "EX", "SE"]
    lesion_auroc      = {}

    for i, name in enumerate(lesion_names):
        if lesion_labels_arr[:, i].sum() > 0:  # skip if no positive samples
            try:
                auroc = roc_auc_score(lesion_labels_arr[:, i],
                                      lesion_probs_arr[:, i])
                lesion_auroc[name] = auroc
            except Exception:
                lesion_auroc[name] = None

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_grade_labels, all_grade_preds,
                          labels=[0, 1, 2, 3, 4])

    avg_loss = total_loss / len(dataloader)

    metrics = {
        "val_loss":      avg_loss,
        "qwk":           qwk,
        "per_class_acc": per_class_acc,
        "lesion_auroc":  lesion_auroc,
        "confusion_matrix": cm,
    }

    return metrics


def print_metrics(metrics: dict, epoch: int = None):
    """Pretty-prints the evaluation metrics."""
    prefix = f"[Epoch {epoch}]" if epoch else "[Eval]"
    print(f"\n{prefix} ─────────────────────────────────────")
    print(f"  Val Loss : {metrics['val_loss']:.4f}")
    print(f"  QWK      : {metrics['qwk']:.4f}  ← primary metric")
    print(f"  Per-class accuracy:")
    for grade, acc in metrics['per_class_acc'].items():
        print(f"    {grade}: {acc:.3f}")
    print(f"  Lesion AUROC:")
    for lesion, auroc in metrics['lesion_auroc'].items():
        val = f"{auroc:.3f}" if auroc is not None else "N/A"
        print(f"    {lesion}: {val}")
    print()