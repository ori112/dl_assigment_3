"""Data loading, quality-control screening, and preprocessing for the
chest X-ray pneumonia dataset.

The dataset ships as folders ``train`` / ``val`` / ``test``, each containing a
``NORMAL`` and a ``PNEUMONIA`` subfolder of JPEG images.  This module turns that
folder layout into clean Python lists, NumPy arrays (for the linear SVM), and
PyTorch DataLoaders (for the convolutional networks).

We use the convention NORMAL = 0, PNEUMONIA = 1 throughout the project.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Label convention used everywhere in the project.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_NORMAL = 0
LABEL_PNEUMONIA = 1


def find_dataset_root(base_dir):
    """Locate the folder that actually contains train/val/test.

    The downloaded archive nests a second ``chest_xray`` folder and also leaves
    a ``__MACOSX`` folder full of junk metadata files, so we walk down from the
    given base directory until we find a folder holding all three splits.
    """
    candidates = [base_dir]
    for current in candidates:
        entries = set(os.listdir(current))
        if {"train", "val", "test"}.issubset(entries):
            return current
        # Descend into any real subfolder (skip the macOS metadata folder).
        for name in entries:
            sub = os.path.join(current, name)
            if os.path.isdir(sub) and name != "__MACOSX":
                candidates.append(sub)
    raise FileNotFoundError(
        f"Could not find a train/val/test layout under {base_dir!r}"
    )


def list_labeled_images(split_dir):
    """Return a list of (image_path, label) pairs for one split folder.

    Files starting with '.' are skipped because the macOS archive sprinkles
    hidden '._' resource-fork files next to the real images.
    """
    items = []
    for class_name in CLASS_NAMES:
        label = CLASS_NAMES.index(class_name)
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.startswith("."):
                continue
            items.append((os.path.join(class_dir, file_name), label))
    return items


def screen_images(items, min_side=32):
    """Quality-control pass over a list of (path, label) pairs.

    Mirrors the dataset authors' description of "screening for quality control
    by removing all low quality or unreadable scans".  We drop any file that:
      * fails to open or whose pixel data is corrupt, or
      * is smaller than ``min_side`` pixels on either edge (too low quality).

    Returns the kept items plus a small report dict for the notebook write-up.
    """
    kept = []
    unreadable = []
    too_small = []
    for path, label in items:
        try:
            # verify() checks the file integrity without fully decoding it,
            # then we must reopen to actually read the size.
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            unreadable.append(path)
            continue
        if width < min_side or height < min_side:
            too_small.append(path)
            continue
        kept.append((path, label))

    report = {
        "total": len(items),
        "kept": len(kept),
        "removed_unreadable": len(unreadable),
        "removed_too_small": len(too_small),
    }
    return kept, report


def count_by_class(items):
    """Helper that counts how many items fall in each class label."""
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in items:
        counts[CLASS_NAMES[label]] += 1
    return counts


def compute_class_weights(items):
    """Inverse-frequency weights for handling the NORMAL/PNEUMONIA imbalance.

    Training has roughly three times as many pneumonia scans as normal ones, so
    weighting the loss by the inverse class frequency stops the network from
    simply favouring the majority class.  Weights are normalized to average 1.
    Returns a float32 NumPy array indexed by class label.
    """
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for _, label in items:
        counts[label] += 1
    inverse = counts.sum() / (len(CLASS_NAMES) * counts)
    return inverse.astype(np.float32)


def make_stratified_val_split(train_items, val_fraction=0.15, seed=0):
    """Carve a stratified validation set out of the training items.

    The released ``val`` folder only holds 16 images, which is far too few to
    tune on, so we hold out a fixed fraction of the training data per class.
    Stratifying keeps the NORMAL / PNEUMONIA ratio the same in both halves.
    """
    rng = random.Random(seed)
    new_train = []
    new_val = []
    for label in range(len(CLASS_NAMES)):
        in_class = [item for item in train_items if item[1] == label]
        rng.shuffle(in_class)
        n_val = int(round(len(in_class) * val_fraction))
        new_val.extend(in_class[:n_val])
        new_train.extend(in_class[n_val:])
    rng.shuffle(new_train)
    rng.shuffle(new_val)
    return new_train, new_val


# ---------------------------------------------------------------------------
# NumPy loader for the linear SVM baseline
# ---------------------------------------------------------------------------

def load_images_as_arrays(items, image_size=64):
    """Decode images into a flat NumPy feature matrix for the SVM.

    Each X-ray is read as grayscale, resized to ``image_size`` x ``image_size``,
    and flattened into a single row.  Pixels are scaled to the [0, 1] range so
    the SVM does not have to cope with raw 0-255 values.

    Returns X of shape (N, image_size * image_size) and y of shape (N,).
    """
    n_samples = len(items)
    n_features = image_size * image_size
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for row, (path, label) in enumerate(items):
        with Image.open(path) as img:
            gray = img.convert("L").resize((image_size, image_size))
        X[row] = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0
        y[row] = label
    return X, y


# ---------------------------------------------------------------------------
# PyTorch pipeline for the convolutional networks
# ---------------------------------------------------------------------------

def compute_mean_std(items, image_size=128, sample_size=500, seed=0):
    """Estimate the grayscale mean and std used to normalize CNN inputs.

    Computing this over a random sample of the training images is plenty
    accurate for normalization and avoids decoding the whole set twice.
    """
    rng = random.Random(seed)
    sample = items if len(items) <= sample_size else rng.sample(items, sample_size)
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    for path, _ in sample:
        with Image.open(path) as img:
            gray = img.convert("L").resize((image_size, image_size))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float((arr ** 2).sum())
        pixel_count += arr.size
    mean = pixel_sum / pixel_count
    # Var = E[x^2] - E[x]^2, then clamp tiny negatives from float error.
    var = max(pixel_sq_sum / pixel_count - mean ** 2, 1e-8)
    std = float(np.sqrt(var))
    return mean, std


class XRayDataset(Dataset):
    """Tiny Dataset that wraps our (path, label) lists.

    Using our own list-backed dataset (instead of torchvision's ImageFolder)
    lets us feed in the re-split train/val partitions we built by hand.
    """

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as img:
            image = img.convert("L")
        return self.transform(image), label


def build_transform(image_size, mean, std, augment=False):
    """Build the preprocessing transform for one channel grayscale X-rays.

    Training uses light augmentation (small rotations / shifts / flips) to
    fight overfitting; validation and test use only resize + normalize.
    """
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        # Chest X-rays are roughly left-right symmetric and may be slightly
        # rotated or off-center, so these augmentations are label-preserving.
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.RandomRotation(degrees=10))
        steps.append(transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=[mean], std=[std]))
    return transforms.Compose(steps)


def make_dataloaders(train_items, val_items, test_items, image_size=128,
                     batch_size=64, mean=None, std=None, augment_train=True,
                     num_workers=0):
    """Build train/val/test DataLoaders plus the normalization stats used.

    Mean/std default to values estimated from the training split so that the
    val and test sets are normalized with the exact same statistics.
    """
    if mean is None or std is None:
        mean, std = compute_mean_std(train_items, image_size=image_size)

    train_tf = build_transform(image_size, mean, std, augment=augment_train)
    eval_tf = build_transform(image_size, mean, std, augment=False)

    train_loader = DataLoader(
        XRayDataset(train_items, train_tf), batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        XRayDataset(val_items, eval_tf), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        XRayDataset(test_items, eval_tf), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, (mean, std)
