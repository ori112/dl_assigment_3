"""A small, reusable training loop for the PyTorch convolutional networks.

Both the two-layer and the deep network are trained through the same
``train_model`` function so the notebook stays short and the two experiments
are directly comparable.
"""

import numpy as np
import torch


def get_device():
    """Pick CUDA when available (e.g. on Colab) and fall back to CPU locally."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_all(model, loader, device):
    """Run the model over a loader and return (y_pred, y_true) as NumPy arrays."""
    model.eval()
    preds = []
    labels = []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Convenience wrapper that returns plain accuracy on a loader."""
    y_pred, y_true = predict_all(model, loader, device)
    return float((y_pred == y_true).mean())


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=10, verbose=True):
    """Train ``model`` and record loss / accuracy history.

    Returns a history dict with:
        'train_loss' : loss at every iteration (for the loss curve)
        'train_acc'  : training accuracy after each epoch
        'val_acc'    : validation accuracy after each epoch
    """
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            history["train_loss"].append(float(loss.item()))

        # End-of-epoch evaluation on both splits.
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"epoch {epoch + 1:2d}/{num_epochs}  "
                  f"loss {history['train_loss'][-1]:.4f}  "
                  f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}")

    return history
