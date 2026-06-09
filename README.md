# Pneumonia Detection from Chest X-Rays

Deep Learning final project (course 99006) — binary classification of chest X-ray images as **Normal** or **Pneumonia**.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,863 JPEG images from pediatric patients at Guangzhou Women and Children's Medical Center, organized into `train / val / test` splits with two classes (`NORMAL`, `PNEUMONIA`).

> The dataset is **not included** in this repository. Download it from Kaggle and place the inner `chest_xray` folder (the one containing `train/`, `val/`, `test/`) anywhere on your machine, then set `DATASET_BASE` in the notebook accordingly.

## Models

| Model | Test Accuracy | NORMAL Recall | PNEUMONIA Recall |
|-------|:------------:|:-------------:|:----------------:|
| Linear SVM (L2) | 74.8 % | 34.2 % | 99.2 % |
| Linear SVM (L1) | 74.5 % | 33.8 % | 99.0 % |
| Two-layer CNN | **86.7 %** | 67.9 % | **97.9 %** |
| Deep CNN (class-weighted) | 82.4 % | **94.4 %** | 75.1 % |

## Project Structure

```
├── pneumonia_project.ipynb   # Main notebook (driver code + written report)
├── data_utils.py             # Dataset loading, QC screening, DataLoaders
├── linear_svm.py             # From-scratch SVM (naive + vectorized, L1/L2)
├── conv_nets.py              # TwoLayerConvNet and DeepConvNet architectures
├── solver.py                 # PyTorch training/evaluation loop
└── metrics.py                # Accuracy, confusion matrix, P/R/F1, plots
```

## Setup

```bash
uv sync          # installs all dependencies (CPU torch build)
uv run jupyter notebook pneumonia_project.ipynb
```

Or open `pneumonia_project.ipynb` directly in **Google Colab** (recommended for GPU training — mount your Drive and set `DATASET_BASE` to the dataset path).

## Requirements

- Python 3.12+
- PyTorch, Torchvision, NumPy, SciPy, Matplotlib, Pillow

No external model weights or third-party implementations are used. All models are trained from scratch.
