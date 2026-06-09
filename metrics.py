"""Evaluation metrics and plotting helpers shared by all of our models.

Kept framework-agnostic on purpose: everything here works on plain NumPy
integer arrays of predictions and labels, so the same functions serve the
NumPy SVM and the PyTorch convolutional networks.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_utils import CLASS_NAMES


def accuracy(y_pred, y_true):
    """Fraction of predictions that match the ground-truth labels."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return float((y_pred == y_true).mean())


def confusion_matrix(y_pred, y_true, num_classes=2):
    """Build the confusion matrix with rows = true class, cols = predicted.

    Entry [i, j] counts samples whose true label is i and predicted label is j.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def per_class_report(y_pred, y_true, num_classes=2):
    """Precision, recall and F1 for each class, computed from the confusion matrix.

    For a medical screening task these per-class numbers matter more than raw
    accuracy: missing a pneumonia case (recall on class 1) is the costly error.
    """
    matrix = confusion_matrix(y_pred, y_true, num_classes)
    report = {}
    for c in range(num_classes):
        true_positive = matrix[c, c]
        predicted_positive = matrix[:, c].sum()
        actual_positive = matrix[c, :].sum()
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        report[CLASS_NAMES[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(actual_positive),
        }
    return report


def plot_confusion_matrix(matrix, title="Confusion matrix", ax=None):
    """Draw a labeled confusion matrix heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    # Annotate each cell with its count, switching text color for readability.
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    return ax


def plot_training_curves(history, title_prefix=""):
    """Plot loss and accuracy curves from a solver history dict.

    Expects keys 'train_loss', and optionally 'train_acc' / 'val_acc'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"{title_prefix} training loss".strip())
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")

    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="train")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title_prefix} accuracy".strip())
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def show_examples(items, n=8, title="Example X-rays"):
    """Display a grid of example images with their class labels.

    Used in the Dataset section of the notebook to give a visual feel for the
    NORMAL vs PNEUMONIA scans.
    """
    from PIL import Image

    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(2 * cols, 2.4))
    for ax, (path, label) in zip(axes, items[:n]):
        with Image.open(path) as img:
            ax.imshow(img.convert("L"), cmap="gray")
        ax.set_title(CLASS_NAMES[label], fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
