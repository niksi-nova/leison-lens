# model.py
# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B4 Multi-Task Model for Diabetic Retinopathy
#
# Architecture:
#   ┌─────────────────────────────────────┐
#   │   EfficientNet-B4 Backbone          │  ← Pretrained on ImageNet
#   │   (feature extractor, frozen stem)  │
#   └──────────────┬──────────────────────┘
#                  │  feature vector [B, 1792]
#          ┌───────┴───────┐
#          ↓               ↓
#   ┌──────────────┐  ┌──────────────────┐
#   │  Grade Head  │  │  Lesion Head     │
#   │  5-class CE  │  │  4-class BCE     │
#   └──────────────┘  └──────────────────┘
#
# Why multi-task?
#   Jointly learning DR grade AND lesion presence acts as a regulariser —
#   the model must learn features that explain both, which are the actual
#   clinically meaningful features (lesions cause grade severity).
#   This is the key novelty of your architecture over vanilla classification.
# ─────────────────────────────────────────────────────────────────────────────

import timm
import torch
import torch.nn as nn
from config import (BACKBONE, NUM_CLASSES, NUM_LESION_CLASSES,
                    DROPOUT_RATE, DEVICE)


class DRMultiTaskModel(nn.Module):
    """
    Multi-task EfficientNet-B4 for:
      - Task 1: DR grade classification (0–4) → 5-class softmax
      - Task 2: Lesion presence detection (MA/HE/EX/SE) → 4-class sigmoid

    The backbone is shared — both tasks learn from the same features.
    """

    def __init__(self):
        super().__init__()

        # ── Backbone: EfficientNet-B4 pretrained on ImageNet ─────────────────
        # num_classes=0  → removes the default 1000-class head
        # global_pool='avg' → applies global average pooling after last conv block
        # Output: feature vector of shape [B, 1792]
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained   = True,
            num_classes  = 0,         # strip classifier
            global_pool  = 'avg'      # keeps spatial → vector compression
        )
        feat_dim = self.backbone.num_features   # 1792 for EfficientNet-B4

        # ── Grade Classification Head ─────────────────────────────────────────
        # Dropout → Linear(1792→256) → ReLU → Linear(256→5)
        # Dropout (0.4) prevents the head from memorising training samples.
        # The 256-unit hidden layer lets the head learn non-linear combinations.
        self.grade_head = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_CLASSES)   # raw logits → CrossEntropyLoss handles softmax
        )

        # ── Lesion Detection Head ─────────────────────────────────────────────
        # Same structure but outputs 4 logits for multi-label classification.
        # BCEWithLogitsLoss applies sigmoid internally (numerically stable).
        self.lesion_head = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_LESION_CLASSES)  # 4 lesion types
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : input tensor [B, 3, 512, 512]

        Returns:
            grade_logits  : [B, 5]  — raw scores for DR grades 0–4
            lesion_logits : [B, 4]  — raw scores for lesion presence
        """
        # Shared feature extraction — both heads see the same representation
        features = self.backbone(x)          # [B, 1792]

        grade_logits  = self.grade_head(features)    # [B, 5]
        lesion_logits = self.lesion_head(features)   # [B, 4]

        return grade_logits, lesion_logits


class MultiTaskLoss(nn.Module):
    """
    Combined loss for DR grade + lesion detection.

    total_loss = CrossEntropy(grade) + λ × BCE(lesion)

    Why separate losses?
    - Grade is ORDINAL (0 < 1 < 2 < 3 < 4) — CrossEntropyLoss treats it as
      multi-class. Note: for better ordinal handling, consider MSE on grade
      as an experiment — that's another ablation row for your paper.
    - Lesions are MULTI-LABEL (multiple can coexist) — BCEWithLogitsLoss
      treats each lesion independently.

    λ (lambda_weight) controls the trade-off. Ablation in train.py.
    """

    def __init__(self, lambda_weight: float = 0.4):
        super().__init__()
        self.lambda_weight  = lambda_weight
        self.ce_loss        = nn.CrossEntropyLoss()
        self.bce_loss       = nn.BCEWithLogitsLoss()

    def forward(self, grade_logits, lesion_logits,
                grade_labels, lesion_labels):
        """
        Args:
            grade_logits   : [B, 5]  model output for grades
            lesion_logits  : [B, 4]  model output for lesions
            grade_labels   : [B]     integer ground truth (0–4)
            lesion_labels  : [B, 4]  binary float ground truth

        Returns:
            total_loss, grade_loss, lesion_loss  (for logging)
        """
        grade_loss  = self.ce_loss(grade_logits, grade_labels)
        lesion_loss = self.bce_loss(lesion_logits, lesion_labels)
        total_loss  = grade_loss + self.lambda_weight * lesion_loss

        return total_loss, grade_loss, lesion_loss


def build_model(lambda_weight: float = 0.4, device: str = DEVICE):
    """
    Convenience function: instantiates model + loss + optimizer together.

    Returns:
        model, criterion, optimizer
    """
    from config import LEARNING_RATE, WEIGHT_DECAY
    import torch.optim as optim

    model     = DRMultiTaskModel().to(device)
    criterion = MultiTaskLoss(lambda_weight=lambda_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )

    # Count trainable parameters — good to log in your paper
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Backbone: {BACKBONE}")
    print(f"[model] Total trainable parameters: {total_params:,}")
    print(f"[model] Lambda (lesion weight): {lambda_weight}")
    print(f"[model] Device: {device}")

    return model, criterion, optimizer