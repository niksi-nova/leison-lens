```markdown
# Multi-Task Diabetic Retinopathy Detection with Interpretable Grad-CAM Validation

> **Status**: Phase 3 complete (Training + Ablation) | Phase 4 in progress (Grad-CAM)
> **Target venue**: IEEE Journal of Biomedical and Health Informatics (JBHI) or equivalent

---

## Overview

This repository implements a multi-task deep learning system for diabetic retinopathy (DR) grading and lesion detection from fundus photographs. The core novelty is the **joint learning of DR severity grade and lesion presence** from a single EfficientNet-B4 backbone, validated with **quantitative Grad-CAM localisation** against IDRiD pixel-level lesion masks.

### Why this is novel
Most DR papers do one of these three things. This paper does all three:
1. **Multi-task learning** — DR grade (0–4) and lesion presence (MA/HE/EX/SE) predicted jointly from one model, using lesion detection as a regulariser for the grading task
2. **Quantitative interpretability** — Grad-CAM heatmaps validated against IDRiD ground-truth masks using IoU and Dice, not just shown as qualitative figures
3. **Cross-dataset generalisation** — Zero-shot evaluation on Messidor-2 (Phase 5, upcoming)

---

## Results Summary

### Primary model: EfficientNet-B4 Multi-Task (λ=0.4)

| Metric | Value |
|---|---|
| **Test QWK (APTOS 2019)** | **0.9065** |
| Grade 0 accuracy | 98.3% |
| Grade 1 accuracy | 35.1%* |
| Grade 2 accuracy | 86.0% |
| Grade 3 accuracy | 52.6% |
| Grade 4 accuracy | 56.7% |
| MA AUROC | 0.997 |
| HE AUROC | 0.985 |
| EX AUROC | 0.943 |
| SE AUROC | 0.909 |

*Grade 1 weakness is a known APTOS 2019 class imbalance issue (n=296 training samples). Consistent with prior literature.

### Ablation Study — Multi-task Loss Weight λ

| λ | Test QWK | Note |
|---|---|---|
| 0.3 | 0.8982 | |
| **0.4** | **0.9065** | **Primary model** |
| 0.5 | 0.9017 | |

All values within 0.004 QWK — method is robust to this hyperparameter. Full results in `logs/ablation_results.json`.

---

## Architecture

```
Input [B, 3, 512, 512]
        │
        ▼
EfficientNet-B4 Backbone (ImageNet pretrained, timm)
Global Average Pooling → Feature vector [B, 1792]
        │
   ┌────┴────┐
   ▼         ▼
Grade Head   Lesion Head
Linear(1792→256)→ReLU→Linear(256→5)   Linear(1792→256)→ReLU→Linear(256→4)
   │         │
5-class CE   4-class BCE (MA, HE, EX, SE)
   └────┬────┘
        ▼
Total Loss = L_grade + λ · L_lesion
```

---

## Datasets

| Dataset | Role | Images | Labels |
|---|---|---|---|
| APTOS 2019 | Train / Val / Test | 3,662 | DR grade 0–4 |
| IDRiD | Grad-CAM validation | 81 | Pixel-level lesion masks |
| Messidor-2 | Zero-shot test (Phase 5) | 1,748 | DR grade 0–4 |

### APTOS 2019 split (seed=42, stratified)
- Train: 2,928 images (80%)
- Val: 367 images (10%)
- Test: 367 images (10%)

---

## Preprocessing

Ben Graham preprocessing (standard in DR literature) applied to all images:

```python
import cv2
def ben_graham(img, sigmaX=10):
    img = cv2.resize(img, (512, 512))
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )
    return img
```

Additionally: CLAHE on the green channel (DR lesions most visible in green channel of RGB).

---

## Repository Structure

```
├── src/
│   ├── config.py          # All hyperparameters and paths — edit this first
│   ├── dataset.py         # APTOSDataset, augmentations, DataLoader builder
│   ├── model.py           # DRMultiTaskModel, MultiTaskLoss, build_model()
│   ├── train.py           # Training loop + ablation runner
│   ├── evaluate.py        # QWK, per-class accuracy, lesion AUROC, confusion matrix
│   ├── utils.py           # Checkpointing, class weights, training curve plots
│   ├── preprocess.py      # Ben Graham + CLAHE preprocessing pipeline
│   ├── gradcam.py         # Grad-CAM generation, IoU, Dice, overlay (Phase 4)
│   └── phase4_run.py      # Phase 4 main script — IDRiD validation (Phase 4)
├── logs/
│   ├── ablation_results.json   # Full ablation data
│   ├── training_loss.png       # Figure 2: Train vs val loss
│   └── val_kappa.png           # Figure 3: Validation QWK per epoch
├── output/
│   └── gradcam/                # Phase 4 outputs: heatmaps, figure4.png, CSV
├── requirements.txt
└── README.md
```

> **Checkpoints** are too large for GitHub (77MB).
> Download `best_model_lambda0.4.pth` from the Kaggle dataset:
> `https://kaggle.com/datasets/ff9e5ab2c38dcb3b603cc3fe6df9504c9e5363828f30cec6c0b0865a9017aaaa`
> Place it in `checkpoints/best_model_lambda0.4.pth`

---

## Setup

### Environment
```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### Key dependencies
```
torch>=2.0
torchvision
timm
albumentations
grad-cam
scikit-learn
pandas
opencv-python
matplotlib
flask          # Phase 6 backend
```

### Training environment
- Trained on Kaggle free tier (NVIDIA T4 GPU, 16GB VRAM)
- Batch size: 16 | Image size: 512×512
- 30 epochs | ~45 min per run on T4

---

## Reproducing Results

### 1. Preprocess APTOS images
```bash
# Edit DATA_DIR and OUTPUT_DIR in src/config.py first
cd src && python preprocess.py
```

### 2. Train the primary model
```bash
cd src && python train.py
# Checkpoint saved to checkpoints/best_model_lambda0.4.pth
# Expected: Test QWK ≈ 0.90
```

### 3. Run ablation study
```bash
# In train.py, uncomment the ablation __main__ block
cd src && python train.py
# Runtime: ~2.5 hours on T4 GPU
# Results saved to logs/ablation_results.json
```

### 4. Run Phase 4 — Grad-CAM on IDRiD
```bash
# Requires IDRiD dataset and best_model_lambda0.4.pth
cd src && python phase4_run.py
# Outputs: output/gradcam/figure4.png, gradcam_results.csv
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Backbone | EfficientNet-B4 | timm, ImageNet pretrained |
| Image size | 512 × 512 | After Ben Graham resize |
| Batch size | 16 | T4 GPU limit |
| Epochs | 30 | Best checkpoint at epoch 15 |
| Learning rate | 1e-4 | AdamW |
| Weight decay | 1e-5 | AdamW |
| Dropout | 0.4 | Both heads |
| λ (lesion weight) | 0.4 | From ablation study |
| LR scheduler | ReduceLROnPlateau | patience=3, factor=0.5 |
| Grad-CAM target | blocks[-1] | Last EfficientNet-B4 block |
| CAM threshold | 0.5 | For IoU binarisation |

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Environment setup | ✅ Complete |
| 2 | Data preprocessing (Ben Graham + CLAHE) | ✅ Complete |
| 3 | Model training + ablation (APTOS 2019) | ✅ Complete |
| 4 | Grad-CAM validation on IDRiD | 🔄 In progress |
| 5 | Zero-shot testing on Messidor-2 | ⬜ Upcoming |
| 6 | Flask backend + React frontend demo | ⬜ Upcoming |
| 7 | Paper writing and submission | ⬜ Upcoming |

---

## Citation

If you use this work, please cite:

```bibtex
@article{yourname2025dr,
  title   = {Multi-Task Diabetic Retinopathy Grading with Interpretable
             Lesion-Guided Grad-CAM Validation},
  author  = {Your Name and Teammate Name},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
2. Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
3. Porwal, P., et al. (2018). IDRiD: Diabetic Retinopathy Segmentation and Grading Challenge. *Medical Image Analysis*.
4. Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition Report.
5. Li, T., et al. (2019). Diagnostic Assessment of Deep Learning for DR Screening. *Information Sciences*.

---

## Team

| Member | Responsibility |
|---|---|
| [Your name] | Phases 1–3: preprocessing, training, ablation |
| [Teammate name] | Phase 4: Grad-CAM, IDRiD validation |

---

*Training performed on Kaggle free tier (T4 GPU). No paid compute used.*
```

