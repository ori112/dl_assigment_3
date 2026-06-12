# Study Guide — Pneumonia X-Ray Classification
### Deep Learning Final Project (99006) — Oral Exam Preparation

---

## Table of Contents
1. [Dataset & Preprocessing](#1-dataset--preprocessing)
2. [Linear SVM — Theory](#2-linear-svm--theory)
3. [Linear SVM — Code (line by line)](#3-linear-svm--code-line-by-line)
4. [Convolutional Networks — Theory](#4-convolutional-networks--theory)
5. [Backpropagation & Automatic Differentiation](#5-backpropagation--automatic-differentiation)
6. [PyTorch Fundamentals](#6-pytorch-fundamentals)
7. [Two-Layer CNN — Architecture & Code](#7-two-layer-cnn--architecture--code)
8. [Deep CNN — Architecture & Code](#8-deep-cnn--architecture--code)
9. [Training Loop — Code (line by line)](#9-training-loop--code-line-by-line)
10. [Metrics — Code (line by line)](#10-metrics--code-line-by-line)
11. [Results & Analysis](#11-results--analysis)
12. [Likely Oral Exam Questions & Answers](#12-likely-oral-exam-questions--answers)

---

## 1. Dataset & Preprocessing

### The dataset
- **Source:** Kermany et al. (Cell, 2018), Guangzhou Women and Children's Medical Center.
- **Size:** 5,863 JPEG chest X-ray images, 2 classes: `NORMAL` (label 0) and `PNEUMONIA` (label 1).
- **Patients:** pediatric, age 1–5. All images are anterior-posterior (AP) views — patient faces the machine.
- **Label quality:** graded independently by two expert physicians; evaluation set reviewed by a third to account for grading errors.
- **Medical context:** Normal lungs appear clear. Bacterial pneumonia shows focal lobar consolidation (white patch in one area). Viral pneumonia shows a diffuse interstitial pattern across both lungs.

### Folder structure
```
chest_xray/
  train/   NORMAL/ (1341 images)   PNEUMONIA/ (3875 images)
  val/     NORMAL/ (8 images)      PNEUMONIA/ (8 images)
  test/    NORMAL/ (234 images)    PNEUMONIA/ (390 images)
```
Total: 5,863 images. The archive also contains a nested duplicate folder and a `__MACOSX/` folder of junk metadata — `find_dataset_root` walks past these automatically.

### Class imbalance
- Train split: **1:2.9 ratio** (NORMAL:PNEUMONIA). Nearly three times as many pneumonia scans.
- This is typical in medical datasets — hospitals over-represent disease cases in archived data.
- **The danger:** a naive classifier that *always* predicts PNEUMONIA gets 390/624 = **62.5% test accuracy** while being medically useless (NORMAL recall = 0%). The SVM's 74.8% is only modestly better.
- **How we handle it:** class-weighted cross-entropy loss for the CNNs (see below).

### Why we re-split the validation set
- The official val folder has only **16 images (8 per class)**. One wrong prediction = 6.25 percentage point swing. This is dominated by noise, not signal.
- **Solution (`make_stratified_val_split`):** hold out a fixed 15% of training data per class.
  - "Stratified" = shuffle each class independently, take first 15% of each. This preserves the 1:2.9 ratio in both halves.
  - Result: 782 val images (201 NORMAL / 581 PNEUMONIA).
- The test set (624 images) is **never touched** until final evaluation.

```python
# data_utils.py — how stratified split works
def make_stratified_val_split(train_items, val_fraction=0.15, seed=0):
    rng = random.Random(seed)
    new_train, new_val = [], []
    for label in range(len(CLASS_NAMES)):           # once for NORMAL, once for PNEUMONIA
        in_class = [item for item in train_items if item[1] == label]
        rng.shuffle(in_class)                       # shuffle within class
        n_val = int(round(len(in_class) * val_fraction))
        new_val.extend(in_class[:n_val])            # first 15% → val
        new_train.extend(in_class[n_val:])          # rest → train
    rng.shuffle(new_train)
    rng.shuffle(new_val)
    return new_train, new_val
```

### Quality-control screening (`screen_images`)
Mirrors the paper: "screened for quality control by removing all low quality or unreadable scans."
```python
def screen_images(items, min_side=32):
    for path, label in items:
        try:
            with Image.open(path) as probe:
                probe.verify()        # checks file integrity without full decode
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            unreadable.append(path)   # corrupt file → drop
            continue
        if width < min_side or height < min_side:
            too_small.append(path)    # degenerate image → drop
            continue
        kept.append((path, label))
```
- Must reopen after `verify()` because PIL's verify() leaves the file in an unusable state.
- In practice: 0 files removed — the published dataset is already clean.

### Image preprocessing pipeline

**Why grayscale?** X-rays are single-channel intensity images. JPEG color channels carry scanner-specific tinting artifacts, not diagnostic information. Dropping them reduces feature dimensionality 3× at zero information cost.

**For the SVM** (`load_images_as_arrays`):
```python
def load_images_as_arrays(items, image_size=64):
    X = np.zeros((n_samples, image_size * image_size), dtype=np.float32)
    for row, (path, label) in enumerate(items):
        with Image.open(path) as img:
            gray = img.convert("L").resize((image_size, image_size))  # L = 8-bit grayscale
        X[row] = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0
```
Steps: decode → grayscale → resize 64×64 → flatten to 4096 → divide by 255 (→ [0,1]).
After loading: subtract training mean (computed only from train, never test — that would be data leakage).

**For the CNNs** (`build_transform` + `compute_mean_std`):
```python
def build_transform(image_size, mean, std, augment=False):
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.RandomRotation(degrees=10))
        steps.append(transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)))
    steps.append(transforms.ToTensor())              # PIL uint8 [0,255] → float32 [0,1]
    steps.append(transforms.Normalize(mean=[mean], std=[std]))  # (x - mean) / std
    return transforms.Compose(steps)
```
- `transforms.Compose` chains transforms: each is applied sequentially to the same image.
- `ToTensor()` converts a PIL image of shape (H, W) to a PyTorch tensor (C, H, W) with values in [0, 1].
- `Normalize` does `(pixel - mean) / std` channel-wise. Training mean ≈ 0.485, std ≈ 0.236.
- Augmentation is **train-only** — val/test always use the `augment=False` path.
- Augmentations used are **label-preserving**: a flipped or slightly rotated chest X-ray still shows the same diagnosis.

**Why normalize?** Without normalization, pixel values in [0, 255] produce large activations in early layers, destabilizing gradient flow. Centering near zero keeps weight updates balanced.

**Why only training statistics?** Using val/test statistics for normalization would be data leakage — the model would have implicitly seen those sets before evaluation.

### Custom Dataset class (`XRayDataset`)
```python
class XRayDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items        # list of (path, label) tuples
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):       # called by DataLoader for each sample
        path, label = self.items[index]
        with Image.open(path) as img:
            image = img.convert("L")   # decode to grayscale here, not upfront
        return self.transform(image), label
```
**Why not use `torchvision.datasets.ImageFolder`?** `ImageFolder` reads from a fixed folder structure. Our re-split creates custom train/val partitions that don't correspond to actual folders on disk — we need to pass in our own `(path, label)` lists. A custom `Dataset` subclass gives us that control.

**How `DataLoader` uses it:** `DataLoader` calls `__getitem__` repeatedly with different indices (shuffled if `shuffle=True`), collects results into batches using `collate_fn`, and returns `(images_tensor, labels_tensor)` pairs.

```python
train_loader = DataLoader(
    XRayDataset(train_items, train_tf),
    batch_size=64,
    shuffle=True,       # shuffle every epoch so the model doesn't learn batch order
    num_workers=0,      # 0 = load in the main process (required on Windows — 
)                       #     multiprocessing with file handles breaks on Windows)
val_loader = DataLoader(
    XRayDataset(val_items, eval_tf),
    batch_size=64,
    shuffle=False,      # val/test: deterministic order so metrics are reproducible
    num_workers=0,
)
```
- **`shuffle=True` for train:** ensures different batches each epoch, preventing the model from memorizing example order.
- **`shuffle=False` for val/test:** evaluation must be deterministic and reproducible.
- **`num_workers=0`:** on Windows, spawning worker processes that hold file handles raises errors. `num_workers=0` runs loading in the main process.

### Class weights (`compute_class_weights`)
```python
def compute_class_weights(items):
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for _, label in items:
        counts[label] += 1
    # Inverse-frequency formula: w_c = N / (C * n_c)
    inverse = counts.sum() / (len(CLASS_NAMES) * counts)
    return inverse.astype(np.float32)
```
Formula: $w_c = \frac{N}{C \cdot n_c}$ where $N$ = total samples, $C$ = 2 classes, $n_c$ = samples in class $c$.

For our training data (1140 NORMAL, 3294 PNEUMONIA, total 4434):
- $w_\text{NORMAL} = 4434 / (2 \times 1140) = 1.944$
- $w_\text{PNEUMONIA} = 4434 / (2 \times 3294) = 0.673$

Passed to `nn.CrossEntropyLoss(weight=tensor([1.944, 0.673]))`: each example's loss is multiplied by its class weight. Misclassifying a NORMAL image costs 1.944× more than misclassifying a PNEUMONIA image, forcing the network to pay attention to the minority class.

---

## 2. Linear SVM — Theory

### What is a linear classifier?
Maps input $x \in \mathbb{R}^D$ to class scores $s \in \mathbb{R}^C$ via $s = Wx$ where $W \in \mathbb{R}^{D \times C}$.
- $D$ = 4096 (64×64 pixels, flattened, +1 bias).
- $C$ = 2 (NORMAL, PNEUMONIA).
- $W_j$ (column $j$) = template for class $j$. The score $s_j = W_j \cdot x$ is the dot product of the template with the image.

### The multiclass SVM hinge loss (Weston-Watkins)

$$L = \frac{1}{N}\sum_{i=1}^{N} \underbrace{\sum_{j \ne y_i} \max\!\left(0,\; s_j - s_{y_i} + \Delta\right)}_{\text{data loss}} + \underbrace{R(W)}_{\text{regularization}}$$

- $y_i$ = correct class for example $i$.
- $s_{y_i} = W_{y_i} \cdot x_i$ = score of the correct class.
- $\Delta = 1$ = margin. We want the correct class to beat every other by at least 1.
- $\max(0, \cdot)$ = hinge: zero loss if a wrong class is already far below the margin; positive loss only when it encroaches.

**Intuition:** "The correct class score must beat all other class scores by a safety margin. Only classes that come too close contribute to the loss."

### Why $\Delta = 1$?
The absolute value of $\Delta$ is arbitrary — the regularization strength $\lambda$ can rescale all scores. Setting $\Delta = 1$ is a convention; the effective constraint is the ratio $\Delta / \lambda$.

### Gradient derivation
For a single example $i$, let $S_i = \{j \ne y_i : s_j - s_{y_i} + \Delta > 0\}$ be the offending classes.

The partial derivative of the data loss w.r.t. weight column $W_j$:

$$\frac{\partial L_i}{\partial W_j} = \begin{cases}
x_i & j \in S_i \quad \text{(wrong class too close — raise its column's score... wait)} \\
-|S_i| \cdot x_i & j = y_i \\
0 & \text{otherwise}
\end{cases}$$

**Why $+x_i$ for offending class $j$?**
$s_j = W_j \cdot x_i$, so $\frac{\partial s_j}{\partial W_j} = x_i$.
The loss for this offending class is $(s_j - s_{y_i} + \Delta)$, so $\frac{\partial L_i}{\partial W_j} = +x_i$.
Gradient descent then does $W_j \leftarrow W_j - \eta \cdot x_i$, which **lowers** $s_j = W_j \cdot x_i$ — moving the wrong class away from the correct one.

**Why $-|S_i| \cdot x_i$ for correct class?**
The correct class score $s_{y_i}$ appears as $-s_{y_i}$ in every offending term, so $\frac{\partial L_i}{\partial W_{y_i}} = -|S_i| \cdot x_i$.
Gradient descent does $W_{y_i} \leftarrow W_{y_i} + \eta \cdot |S_i| \cdot x_i$, which **raises** $s_{y_i}$ — pushing the correct class up.

### L2 vs L1 Regularization

| Property | L2: $\lambda \|W\|_F^2 = \lambda\sum_{ij} W_{ij}^2$ | L1: $\lambda \|W\|_1 = \lambda\sum_{ij} \|W_{ij}\|$ |
|----------|------------------------------------------------------|------------------------------------------------------|
| Gradient | $2\lambda W$ | $\lambda \cdot \text{sign}(W)$ |
| Effect | Shrinks all weights toward 0 smoothly | Drives many weights exactly to 0 (sparsity) |
| Differentiable | Yes, everywhere | No, at $W_{ij}=0$ (use subgradient: sign(0)=0) |
| Unique solution | Yes | Possibly multiple (non-strictly convex) |

**In our results:** L1 (74.5%) ≈ L2 (74.8%). The bottleneck is the representational capacity of raw pixels — neither regularizer can fix a feature space that's insufficient for linear separation.

### The bias trick
Instead of $s = Wx + b$, append a constant 1 to each input:
$$\tilde{x} = [x_1, \ldots, x_D, 1], \quad \tilde{W} \in \mathbb{R}^{(D+1) \times C}$$
Then $\tilde{W}^T \tilde{x} = Wx + b$ where the last row of $\tilde{W}$ plays the role of $b$. One matrix to optimize instead of two.

### Numerical gradient check (central difference)
$$\frac{\partial L}{\partial W_{ij}} \approx \frac{L(W_{ij}+h) - L(W_{ij}-h)}{2h}, \quad h = 10^{-5}$$

Relative error: $\frac{|g_\text{analytic} - g_\text{numeric}|}{|g_\text{analytic}| + |g_\text{numeric}| + \epsilon}$

If relative error < $10^{-7}$, gradient is correct. Our result: $7 \times 10^{-11}$ — passes comfortably.

**Why central difference and not forward difference?** Forward difference error is $O(h)$; central difference error is $O(h^2)$ — much more accurate for the same $h$.

### Mini-batch SGD
- Full gradient over 4434 images per step is expensive.
- Sample 200 random examples (with replacement), compute gradient on that mini-batch.
- 1500 iterations × 200 batch = each example seen ~68 times on average.
- Update: $W \leftarrow W - \eta \cdot \hat{g}$ where $\hat{g}$ is the mini-batch gradient estimate.
- The noise in the estimate can actually help escape shallow local minima.

---

## 3. Linear SVM — Code (line by line)

### `svm_loss_naive` (`linear_svm.py:15`)

```python
def svm_loss_naive(W, X, y, reg, reg_type="l2", delta=1.0):
    num_train = X.shape[0]       # N examples
    num_classes = W.shape[1]     # C = 2
    loss = 0.0
    dW = np.zeros_like(W)        # gradient accumulator, shape (D, C)

    for i in range(num_train):
        scores = X[i].dot(W)               # (C,) — one score per class
        correct_class_score = scores[y[i]] # scalar — correct class score

        for j in range(num_classes):
            if j == y[i]:
                continue           # skip the correct class
            margin = scores[j] - correct_class_score + delta
            if margin > 0:         # this class violates the margin
                loss += margin
                dW[:, j] += X[i]           # gradient for wrong class j
                dW[:, y[i]] -= X[i]        # gradient for correct class (once per offender)

    loss /= num_train              # average over examples
    dW /= num_train

    loss, dW = _add_regularization(loss, dW, W, reg, reg_type)
    return loss, dW
```

Key point: `dW[:, y[i]] -= X[i]` is accumulated inside the inner loop, so the correct class column gets $-|S_i| \cdot x_i$ in total across all offending $j$.

### `svm_loss_vectorized` (`linear_svm.py:55`) — step by step

```python
scores = X.dot(W)
# scores: (N, C) — all class scores for all examples at once
```
```python
correct_class_scores = scores[np.arange(N), y].reshape(-1, 1)
# np.arange(N): [0,1,2,...,N-1] — row indices
# y: [y_0, y_1, ..., y_{N-1}] — column indices (correct class per example)
# scores[np.arange(N), y]: fancy indexing → (N,) correct scores
# .reshape(-1,1): (N,1) for broadcasting in the next step
```
```python
margins = np.maximum(0.0, scores - correct_class_scores + delta)
# Broadcasting: (N,C) - (N,1) → (N,C) — subtract correct score from each column
# np.maximum: apply the hinge (clamp to 0)
```
```python
margins[np.arange(N), y] = 0.0
# Zero out the correct-class column (the j==y_i case we skip in the naive version)
```
```python
loss = margins.sum() / N
# Sum all margin violations, average over examples
```
```python
indicator = (margins > 0).astype(W.dtype)   # (N, C): 1 where margin > 0, else 0
indicator[np.arange(N), y] = -indicator.sum(axis=1)
# For each example i, the correct class gets -|S_i| (negative count of offenders)
# This perfectly encodes the gradient formula: +1 for offenders, -|S_i| for correct
dW = X.T.dot(indicator) / N
# X.T: (D, N), indicator: (N, C) → product: (D, C) = dW
# This is a matrix multiplication that simultaneously accumulates x_i contributions
# for all examples
```

The indicator matrix is the vectorized version of the per-example loop accumulation. Every `1` in position `(i,j)` contributes `X[i]` to `dW[:,j]`; every `-|S_i|` in position `(i, y_i)` contributes `-|S_i|*X[i]` to `dW[:,y_i]`.

### `_add_regularization` (`linear_svm.py:77`)
```python
def _add_regularization(loss, dW, W, reg, reg_type):
    if reg_type == "l2":
        loss += reg * np.sum(W * W)   # ||W||_F^2
        dW += 2 * reg * W             # gradient of ||W||^2 is 2W
    elif reg_type == "l1":
        loss += reg * np.sum(np.abs(W))
        dW += reg * np.sign(W)        # subgradient of |w| is sign(w)
```

### `LinearClassifier.train` (`linear_svm.py:130`)
```python
def train(self, X, y, learning_rate, reg, reg_type, num_iters, batch_size, seed):
    X = self._append_bias(X)          # append column of 1s → (N, D+1)
    rng = np.random.RandomState(seed)
    self.W = 0.001 * rng.randn(dim, num_classes)  # small init → near-zero initial scores

    for it in range(num_iters):
        batch_idx = rng.choice(num_train, batch_size, replace=True)  # sample with replacement
        X_batch = X[batch_idx]         # (batch_size, D+1)
        y_batch = y[batch_idx]         # (batch_size,)

        loss, dW = svm_loss_vectorized(self.W, X_batch, y_batch, reg, reg_type)
        self.W -= learning_rate * dW   # gradient descent step
```

**Why small random init (0.001)?** We want initial scores near zero so all classes start roughly equal. Large random init would immediately produce large margins, making gradients initially very large or very sparse.

**Why `replace=True` in sampling?** Standard mini-batch SGD samples with replacement (theoretically equivalent to sampling without for large datasets). In practice this means some examples appear in a batch twice and some not at all — this is fine and simpler to implement.

---

## 4. Convolutional Networks — Theory

### What is a 2D convolution?
A learnable filter $K$ of shape $(k, k)$ slides across the input feature map, computing a dot product at each position:

$$(\text{output})_{x,y} = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} K_{i,j} \cdot \text{input}_{x+i,\; y+j}$$

With $F$ filters, the output has $F$ channels (one feature map per filter). Each filter learns to detect a specific pattern (edge orientation, blob, texture).

**Why padding=1 for a 3×3 kernel?**
Without padding, a 3×3 conv on a $H \times W$ input gives $(H-2) \times (W-2)$ output — the spatial size shrinks. With `padding=1` (one zero-pixel border), the output size is exactly $H \times W$. Formula: $H_\text{out} = H_\text{in} + 2p - k + 1 = H + 2\cdot1 - 3 + 1 = H$.

**Weight sharing:** the same $k \times k$ filter weights are used at every spatial location. A conv layer with $F$ filters and $C_\text{in}$ input channels has $F \times (C_\text{in} \times k^2 + 1)$ parameters — independent of image size. A fully-connected layer mapping the same sized input to the same output would need billions of parameters.

**Parameter count example (Block 1 of DeepConvNet):**
- Input channels: 1, Output channels: 32, kernel 3×3
- Parameters: $32 \times (1 \times 3 \times 3 + 1) = 32 \times 10 = 320$

### ReLU
$$\text{ReLU}(x) = \max(0, x)$$
- Introduces non-linearity. Without it, stacking linear layers is still linear: $W_2(W_1 x) = (W_2 W_1)x$.
- Computationally trivial (threshold at 0).
- **Vanishing gradient advantage:** sigmoid/tanh saturate (gradient → 0 for large |x|). ReLU gradient is 1 for x > 0, so gradients flow unchanged through active neurons.
- **Dead neurons:** if a neuron's input is always negative, its gradient is always 0 and it never updates ("dead ReLU"). Not a major issue here with BN.

### MaxPool2d(2)
Divides the feature map into non-overlapping 2×2 windows, keeps the maximum:
- Output size: $H/2 \times W/2$ (halves spatial dimensions).
- **No learned parameters** — purely a downsampling operation.
- **Translation invariance:** if a feature (e.g. a bright opacification) shifts by 1 pixel, the max-pool output is unchanged as long as it stays in the same 2×2 window.
- **Why downsample?** Reduces computation in later layers and forces the network to build increasingly global representations.
- Spatial progression in DeepConvNet: 128 → 64 → 32 → 16 after 3 maxpool blocks.

### Batch Normalization (BatchNorm2d)
For a convolutional layer with $C$ channels, BN normalizes independently per channel across the batch and spatial dimensions:

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

where $\mu_c = \frac{1}{NHW}\sum_{n,h,w} x_{n,c,h,w}$ is the mean over batch, height, and width for channel $c$.

Then: $y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$ where $\gamma_c, \beta_c$ are **learnable per-channel** parameters.

**Why it helps (4 reasons):**
1. **Stable activations:** every layer starts with roughly zero-mean, unit-variance activations regardless of what the previous layer learned. Training is faster.
2. **Higher learning rates:** without BN, large learning rates cause activations to explode. BN clips this.
3. **Implicit regularization:** the batch-level statistics introduce noise that acts like a regularizer (similar to dropout).
4. **Reduces covariate shift:** different X-ray scanners produce images with different brightness/contrast. BN normalizes these systematic differences at every layer.

**Train vs eval mode (critical):**
- **During training:** $\mu_c$ and $\sigma_c^2$ are computed from the *current mini-batch*. BN also accumulates a running exponential moving average: `running_mean = 0.9 * running_mean + 0.1 * batch_mean`.
- **During evaluation:** uses the *fixed running mean/std* accumulated during training — not the batch statistics. This gives deterministic, stable predictions for a single image.
- **PyTorch:** `model.train()` activates batch statistics; `model.eval()` switches to running statistics. Forgetting `.eval()` is a common bug.

### Dropout
`Dropout(p=0.5)` randomly zeroes each neuron output with probability $p$ during each forward pass in training.

**Why it regularizes:**
- No single neuron can be relied upon → network learns redundant, distributed representations.
- Equivalent to training $2^n$ sub-networks (one for each binary mask) and averaging predictions.
- During evaluation (`model.eval()`): all neurons are active. To preserve expected activation magnitude, outputs are scaled by $1/(1-p)$ during training (PyTorch's "inverted dropout" — so no scaling is needed at test time).

**Where we apply it:** only in the classifier head (after the conv features), not in the conv blocks. The conv blocks use BN as their regularizer; dropout in conv blocks can hurt performance.

### Adaptive Average Pooling (`AdaptiveAvgPool2d`)
`AdaptiveAvgPool2d((4, 4))` maps the feature map (after 3 maxpool blocks: 128→64→32→16) to a fixed **4×4 spatial grid** by computing averages over variable-size windows.

```
After 3 MaxPool blocks with input 128: spatial size = 128/2/2/2 = 16
AdaptiveAvgPool2d(4,4): averages 16×16 → 4×4 (window size 4×4)
Flat output: 128 channels × 4 × 4 = 2048 features
```

**Why useful:** the `Linear` head needs a fixed input size. Without adaptive pooling, changing the input resolution would require changing `Linear(2048, 256)` to `Linear(new_flat_dim, 256)`. With adaptive pooling, the architecture works for any input resolution.

### Cross-entropy loss and logits
The network outputs **logits** $s \in \mathbb{R}^C$ — raw, unnormalized scores (not probabilities). The loss applies softmax internally:

$$p_j = \frac{e^{s_j}}{\sum_k e^{s_k}} \quad \text{(softmax)}$$

$$L_i = -\log p_{y_i} = -s_{y_i} + \log\sum_k e^{s_k} \quad \text{(cross-entropy)}$$

PyTorch's `nn.CrossEntropyLoss` combines softmax + negative log-likelihood in one numerically stable operation. **Do not apply softmax before passing to CrossEntropyLoss** — it would be applied twice.

**What are logits?** The term comes from log-odds. They are the raw scores before any normalization. A positive logit for class $j$ means the model leans toward class $j$; the argmax of logits gives the predicted class.

**With class weights:** each example's loss is multiplied by $w_{y_i}$:
$$L = \frac{1}{N}\sum_i w_{y_i} \cdot (-\log p_{y_i})$$

### Adam optimizer
Improves on SGD by maintaining per-parameter adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(momentum — 1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(2nd moment — variance)}$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t) \quad \text{(bias correction)}$$
$$W \leftarrow W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Parameters: $\eta=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

**Why better than SGD?** Parameters with historically large gradients get a small effective learning rate (safe); parameters with small gradients get a large effective rate (exploratory). No manual learning rate tuning per layer.

**Weight decay:** we pass `weight_decay=1e-4` to Adam. This adds L2 regularization on the weights directly to the update:
$$W \leftarrow W - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} - \eta \lambda W$$
where $\lambda = 10^{-4}$. Prevents weights from growing too large (different from the class-weighted loss — this regularizes the weights themselves, not the loss per example).

---

## 5. Backpropagation & Automatic Differentiation

### What is backpropagation?
The algorithm for computing $\frac{\partial L}{\partial \theta}$ for every parameter $\theta$ in the network efficiently. It applies the **chain rule** backward through the computation graph.

**Chain rule:** if $L = f(g(\theta))$, then $\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial \theta}$.

In a network: $L = \text{loss}(\text{softmax}(W_2 \cdot \text{ReLU}(W_1 x)))$

Backprop computes gradients starting from $\frac{\partial L}{\partial L}=1$ and propagating backward:
$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial W_2}$$
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial h} \cdot \frac{\partial h}{\partial W_1}$$
where $h$ is the hidden layer and $s$ the final scores.

### How PyTorch does it (`autograd`)
PyTorch builds a **dynamic computation graph** during the forward pass. Every tensor operation records its inputs and the function used. When `loss.backward()` is called:
1. Starts from `loss` (scalar).
2. Traverses the graph in reverse (topological order).
3. At each node, computes the local gradient using the stored operation.
4. Multiplies by the incoming gradient (chain rule) and accumulates into `.grad`.

```python
logits = model(images)      # forward: builds computation graph
loss = criterion(logits, targets)
loss.backward()             # backward: traverses graph, fills .grad for all parameters
optimizer.step()            # uses .grad to update parameters
optimizer.zero_grad()       # MUST clear .grad before next backward (PyTorch accumulates)
```

**`@torch.no_grad()`:** wraps a function so that no computation graph is built. Used for inference — saves memory (no graph stored) and speeds up computation. The outputs cannot have `.backward()` called on them.

```python
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)               # no graph built — faster, less memory
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(labels)
```

`logits.argmax(dim=1)` returns the index of the maximum logit along the class dimension — that's the predicted class. No softmax needed since softmax is monotonic (argmax is unchanged).

---

## 6. PyTorch Fundamentals

### `nn.Module`
The base class for all neural network layers and models in PyTorch.

```python
class TwoLayerConvNet(nn.Module):   # inherit from nn.Module
    def __init__(self, ...):
        super().__init__()          # MUST call parent init — registers parameter tracking
        self.conv = nn.Conv2d(...)  # assigning nn.Module or nn.Parameter → auto-registered
        self.fc1 = nn.Linear(...)

    def forward(self, x):           # defines computation; called when you do model(x)
        x = self.pool(self.relu(self.conv(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc2(self.relu(self.fc1(x)))
```

**Key methods:**
- `model.parameters()` — returns all registered learnable parameters (used by optimizer).
- `model.train()` — sets training mode (BN uses batch stats, Dropout active).
- `model.eval()` — sets eval mode (BN uses running stats, Dropout disabled).
- `model.to(device)` — moves all parameters and buffers to CPU/CUDA.

**Why inherit from `nn.Module`?** PyTorch auto-tracks all `nn.Parameter` and sub-`nn.Module` objects assigned as attributes. This enables `model.parameters()`, `model.state_dict()`, `.to(device)`, `.train()/.eval()` to work automatically.

### Tensors and devices
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- On this machine: CPU (Intel Iris Xe has no CUDA). On Colab with GPU runtime: CUDA.
- Data and model must be on the **same device**. `images.to(device)` and `model.to(device)` move them.
- `.cpu().numpy()` — must move to CPU before converting to NumPy (CUDA tensors can't be directly converted).

### Tensor dimensions convention
PyTorch uses **(batch, channels, height, width)** — NCHW format.
- Grayscale X-ray after `ToTensor()`: `(1, 128, 128)` — 1 channel (grayscale).
- After loading a batch of 64: `(64, 1, 128, 128)`.
- After `Conv2d(1, 16, 3)`: `(64, 16, 128, 128)` — 16 feature maps.
- After `MaxPool2d(2)`: `(64, 16, 64, 64)`.
- After `Flatten(start_dim=1)`: `(64, 65536)` — preserve batch dim, flatten the rest.

### `nn.Sequential`
Chains modules so the output of each is the input of the next:
```python
self.features = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # ...
)
```
`self.features(x)` applies all layers in order. Used in `DeepConvNet` to build the conv blocks programmatically from the `channels` tuple.

---

## 7. Two-Layer CNN — Architecture & Code

### Full architecture with tensor shapes (`conv_nets.py:16`)
```python
class TwoLayerConvNet(nn.Module):
    def __init__(self, in_channels=1, img_size=128, num_filters=16,
                 hidden_dim=100, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        # padding=1 preserves spatial size: 128 → 128
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        # MaxPool(2): spatial size halves: 128 → 64

        pooled_size = img_size // 2           # 128 // 2 = 64
        flat_dim = num_filters * pooled_size * pooled_size  # 16 * 64 * 64 = 65536

        self.fc1 = nn.Linear(flat_dim, hidden_dim)   # 65536 → 256
        self.fc2 = nn.Linear(hidden_dim, num_classes) # 256 → 2

    def forward(self, x):
        # x: (batch, 1, 128, 128)
        x = self.pool(self.relu(self.conv(x)))  # → (batch, 16, 64, 64)
        x = torch.flatten(x, start_dim=1)       # → (batch, 65536)
        x = self.relu(self.fc1(x))              # → (batch, 256)
        return self.fc2(x)                      # → (batch, 2)  [logits]
```

**Parameter count:**
- `conv`: 16 filters × (1 in-channel × 3×3 kernel + 1 bias) = **160**
- `fc1`: 65536 × 256 + 256 bias = **16,777,472** — the dominant cost
- `fc2`: 256 × 2 + 2 bias = **514**
- **Total: ~16.8M parameters** — mostly in fc1 because we don't pool aggressively

**Why is fc1 so large?** After just one maxpool, the feature map is still 64×64. With 16 channels: 16×64×64=65536 features. No amount of clever design reduces this without more pooling.

### Sanity check: overfit a tiny batch
Before full training, run 40 epochs on 32 samples, verify train_acc → 1.0:
- If this fails → bug in forward pass, loss, or optimizer.
- If it passes → gradients flow correctly; proceed to full training.
- Result: train_acc = 1.0000 after 40 epochs — sanity check passed.

### Full training results
- **20 epochs**, Adam lr=1e-3, weight_decay=1e-4, batch=64, augmented training set.
- Test accuracy: **86.7%** (+11.9% over SVM)
- NORMAL: precision=0.952, recall=0.679, F1=0.793
- PNEUMONIA: precision=0.836, recall=0.979, F1=0.902
- **Interpretation:** a single conv stage nearly doubles NORMAL recall vs SVM (34%→68%). The conv filter learns local opacification texture patterns that are linearly inseparable in raw pixel space.

---

## 8. Deep CNN — Architecture & Code

### Full architecture with tensor shapes (`conv_nets.py:44`)
```
Input:  (batch, 1, 128, 128)

Block 1: Conv2d(1→32, 3×3, pad=1) → (batch, 32, 128, 128)
         BatchNorm2d(32)
         ReLU
         MaxPool2d(2)             → (batch, 32, 64, 64)

Block 2: Conv2d(32→64, 3×3, pad=1) → (batch, 64, 64, 64)
         BatchNorm2d(64)
         ReLU
         MaxPool2d(2)             → (batch, 64, 32, 32)

Block 3: Conv2d(64→128, 3×3, pad=1) → (batch, 128, 32, 32)
         BatchNorm2d(128)
         ReLU
         MaxPool2d(2)             → (batch, 128, 16, 16)

AdaptiveAvgPool2d(4,4)            → (batch, 128, 4, 4)
Flatten                           → (batch, 2048)
Linear(2048→256) → ReLU
Dropout(0.5)
Linear(256→2)                     → (batch, 2)  [logits]
```

### How the blocks are built in code
```python
blocks = []
prev_channels = in_channels    # starts at 1
for out_channels in channels:  # iterates (32, 64, 128)
    blocks.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1))
    blocks.append(nn.BatchNorm2d(out_channels))
    blocks.append(nn.ReLU())
    blocks.append(nn.MaxPool2d(kernel_size=2))
    prev_channels = out_channels   # next block's input = this block's output
self.features = nn.Sequential(*blocks)  # *blocks unpacks list into positional args
```
The `channels=(32, 64, 128)` tuple fully controls depth and width. Passing `channels=(16,32,64)` gives a smaller network.

### Why channels double (32→64→128)?
Each MaxPool halves the spatial resolution: 128→64→32→16. The number of feature maps doubles to compensate, keeping total representational capacity roughly constant. This follows VGG/ResNet design principles.

### Parameter count (approximate)
- Block 1: 32×(1×9+1) = 320
- Block 2: 64×(32×9+1) = 18,496
- Block 3: 128×(64×9+1) = 73,856
- BN layers: 4 params/channel (γ, β, running_mean, running_var — γ, β are trainable)
- FC1: 2048×256+256 = 524,544
- FC2: 256×2+2 = 514
- **Total: ~617K** — vs 16.8M for TwoLayerConvNet, because AdaptiveAvgPool2d collapses the spatial dims before the FC layer

### Class-weighted loss code
```python
class_weights = du.compute_class_weights(train_items)  # [1.944, 0.673] float32
weighted_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights).to(device)   # must be on same device as model
)
```
Effect: NORMAL errors cost 1.944× more, PNEUMONIA errors cost 0.673×. The network adjusts its decision threshold accordingly.

### Full training results
- **30 epochs**, Adam lr=1e-3, weight_decay=1e-4, class-weighted loss, batch=64.
- Test accuracy: **82.4%**
- NORMAL: precision=0.695, recall=0.944, F1=0.801
- PNEUMONIA: precision=0.958, recall=0.751, F1=0.842
- **Key insight:** lower overall accuracy than TwoLayerConvNet (82.4% vs 86.7%) because class weighting deliberately penalizes NORMAL errors more, shifting the boundary. NORMAL recall jumps from 67.9% → 94.4%.

---

## 9. Training Loop — Code (line by line)

### `train_model` (`solver.py:38`)
```python
def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, num_epochs=10, verbose=True):
    model.to(device)             # move all model parameters to GPU/CPU
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()            # IMPORTANT: activate dropout + BN batch stats

        for images, targets in train_loader:
            images = images.to(device)    # move batch to GPU/CPU
            targets = targets.to(device)

            optimizer.zero_grad()         # clear accumulated gradients
            logits = model(images)        # forward pass → (batch, 2)
            loss = criterion(logits, targets)  # cross-entropy loss (scalar)
            loss.backward()              # backprop: fills .grad for all parameters
            optimizer.step()             # update W ← W - lr * grad (Adam step)

            history["train_loss"].append(float(loss.item()))
            # .item() converts a 1-element tensor to a Python float

        # Evaluate after every epoch (no gradient computation needed)
        model.eval()             # IMPORTANT: disable dropout + switch BN to running stats
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc   = evaluate_accuracy(model, val_loader, device)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    return history
```

**Common bugs this code avoids:**
1. `optimizer.zero_grad()` before backward — without this, gradients accumulate across batches.
2. `model.train()` at the start of the training loop (not just once at the beginning).
3. `model.eval()` before evaluation — forgetting this gives wrong val/test numbers with BN/Dropout.
4. `images.to(device)` AND `targets.to(device)` — both must be on the same device as the model.

### `evaluate_accuracy` and `predict_all`
```python
@torch.no_grad()         # decorator: disables autograd for the entire function
def predict_all(model, loader, device):
    model.eval()         # (caller already sets this, but safe to set again)
    preds, labels = [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)              # (batch, 2)
        preds.append(logits.argmax(dim=1).cpu().numpy())  # argmax along class dim
        labels.append(targets.numpy())      # targets are already on CPU
    return np.concatenate(preds), np.concatenate(labels)

def evaluate_accuracy(model, loader, device):
    y_pred, y_true = predict_all(model, loader, device)
    return float((y_pred == y_true).mean())
```

`logits.argmax(dim=1)`: for tensor of shape `(batch, 2)`, `dim=1` means "find the max along the class axis" — returns shape `(batch,)` with values 0 or 1. No softmax needed because softmax is monotonic (argmax is unchanged by it).

---

## 10. Metrics — Code (line by line)

### `confusion_matrix` (`metrics.py:22`)
```python
def confusion_matrix(y_pred, y_true, num_classes=2):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix
```
**Convention:** rows = true label, columns = predicted label.
- `matrix[0, 0]` = correctly predicted NORMAL (true=0, pred=0)
- `matrix[1, 1]` = correctly predicted PNEUMONIA
- `matrix[0, 1]` = NORMAL predicted as PNEUMONIA (false positive for PNEUMONIA)
- `matrix[1, 0]` = PNEUMONIA predicted as NORMAL (false negative — **the dangerous error**)

### `per_class_report` (`metrics.py:34`)
```python
def per_class_report(y_pred, y_true, num_classes=2):
    matrix = confusion_matrix(y_pred, y_true, num_classes)
    for c in range(num_classes):
        true_positive  = matrix[c, c]          # correctly predicted class c
        predicted_positive = matrix[:, c].sum() # all predictions of class c (column sum)
        actual_positive    = matrix[c, :].sum() # all true instances of class c (row sum)

        precision = true_positive / predicted_positive  # of all I predicted c, how many are c
        recall    = true_positive / actual_positive     # of all true c, how many did I catch
        f1 = 2 * precision * recall / (precision + recall)
```

**Precision formula:** $P_c = \frac{TP_c}{\text{column sum}_c} = \frac{\text{matrix}[c,c]}{\sum_r \text{matrix}[r,c]}$

**Recall formula:** $R_c = \frac{TP_c}{\text{row sum}_c} = \frac{\text{matrix}[c,c]}{\sum_k \text{matrix}[c,k]}$

**F1 formula:** harmonic mean of P and R: $F1 = \frac{2PR}{P+R}$. Harmonic mean penalizes heavily when either is low — a model with P=1.0, R=0.01 gets F1=0.02, not 0.505.

### Full per-class metrics (all four models)

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| SVM L2 | NORMAL | 0.964 | 0.342 | 0.505 |
| SVM L2 | PNEUMONIA | 0.715 | 0.992 | 0.831 |
| SVM L1 | NORMAL | 0.952 | 0.338 | 0.498 |
| SVM L1 | PNEUMONIA | 0.713 | 0.990 | 0.829 |
| 2-layer CNN | NORMAL | 0.952 | 0.679 | 0.793 |
| 2-layer CNN | PNEUMONIA | 0.836 | 0.979 | 0.902 |
| Deep CNN | NORMAL | 0.695 | 0.944 | 0.801 |
| Deep CNN | PNEUMONIA | 0.958 | 0.751 | 0.842 |

**Reading the SVM L2 row:**
- NORMAL precision 0.964: when the SVM predicts NORMAL, it's right 96.4% of the time. But...
- NORMAL recall 0.342: it only *predicts* NORMAL for 34.2% of actually-normal patients. The other 65.8% are misclassified as PNEUMONIA.
- High PNEUMONIA recall 0.992: almost never misses a pneumonia case — because it nearly always predicts PNEUMONIA.

### `accuracy` (`metrics.py:14`)
```python
def accuracy(y_pred, y_true):
    return float((np.asarray(y_pred) == np.asarray(y_true)).mean())
```
Fraction of exactly correct predictions. Simple, but misleading on imbalanced data — use with confusion matrix.

---

## 11. Results & Analysis

### Summary

| Model | Test Acc | NORMAL Recall | PNEUMONIA Recall |
|-------|:--------:|:-------------:|:----------------:|
| Linear SVM (L2) | 74.8% | 34.2% | 99.2% |
| Linear SVM (L1) | 74.5% | 33.8% | 99.0% |
| Two-layer CNN | **86.7%** | 67.9% | **97.9%** |
| Deep CNN (weighted) | 82.4% | **94.4%** | 75.1% |

### Why SVM test accuracy is 74.8% despite val accuracy 95.9%
- **Distribution shift:** re-split val is 74.3% PNEUMONIA (581/782); original test is 62.5% PNEUMONIA (390/624). The test set has proportionally more NORMAL cases.
- **Near-degenerate SVM:** the model nearly always predicts PNEUMONIA. This scores well on a PNEUMONIA-heavy val set, less well on the more balanced test set.
- **Overfitting the sweep:** with only 782 val images, the best (lr, reg) pair by chance looks good on val but doesn't generalise.
- A model that always predicts PNEUMONIA scores: val acc = 74.3%, test acc = 62.5%. Our SVM (74.8% test) is only marginally above this floor.

### Why L1 ≈ L2 for SVM
74.8% vs 74.5% — negligible difference. The bottleneck is expressive capacity (linear model on raw pixels), not overfitting. No regularization can fix a feature space that's intrinsically insufficient for linear separation.

### Why Deep CNN accuracy < Two-layer CNN accuracy
The Deep CNN uses **class-weighted loss** (NORMAL weight 1.94×). This does not make the model more powerful — it changes *what the model optimizes for*.

Effect on the decision boundary: the model is penalized 1.94× more for missing a NORMAL case. It responds by predicting PNEUMONIA less aggressively — any borderline case that could be NORMAL gets classified as NORMAL.

Result:
- NORMAL recall: 67.9% → **94.4%** (catches almost all healthy patients)
- PNEUMONIA recall: 97.9% → **75.1%** (misses more pneumonia cases)
- Overall accuracy drops because test set has more PNEUMONIA (390 vs 234), and we now miss 25% of them.

**This is intentional.** For a secondary screening tool, high NORMAL recall matters more.

### Clinical interpretation of recall tradeoffs
For pneumonia:
- **Miss a sick patient (low PNEUMONIA recall):** patient goes untreated → potentially fatal. *Type II error.*
- **Flag a healthy patient (low NORMAL recall):** unnecessary antibiotics, patient anxiety, cost. *Type I error.*

For mass screening (first-line triage) → maximize PNEUMONIA recall → **Two-layer CNN (97.9%)**.
For antibiotics stewardship (reduce unnecessary prescriptions) → maximize NORMAL recall → **Deep CNN (94.4%)**.

### Reading the confusion matrix
```
Convention: rows = TRUE label, columns = PREDICTED label

                  Predicted NORMAL   Predicted PNEUMONIA
True NORMAL       matrix[0,0]  (TP)   matrix[0,1]  (FP)
True PNEUMONIA    matrix[1,0]  (FN)   matrix[1,1]  (TP)
```
- Recall for class c = `matrix[c,c] / matrix[c,:].sum()` (row sum = all true positives)
- Precision for class c = `matrix[c,c] / matrix[:,c].sum()` (col sum = all predicted positives)
- For SVM: `matrix[0,1]` (NORMAL flagged as PNEUMONIA) is very large — ~66% of healthy patients.
- The most dangerous cell is `matrix[1,0]` (PNEUMONIA missed) — sick patient gets no treatment.

### Why val accuracy oscillates epoch to epoch
Both CNN validation curves show significant oscillation (e.g. 83%→94%→86%→95%). Root cause: only 782 val images. A single bad batch prediction can swing val accuracy by ~0.1%. With a larger dataset or k-fold cross-validation, the curves would be smoother and more informative.

---

## 12. Likely Oral Exam Questions & Answers

### Dataset & Preprocessing

**Q: Why did you re-split the validation set?**
A: The official val set has 16 images — 8 per class. One misclassification is a 6.25 percentage point swing. That's noise, not signal. We carved a stratified 15% hold-out from training (782 images) to get a reliable tuning signal.

**Q: What does stratified mean and why does it matter here?**
A: We sample 15% independently from each class so the NORMAL:PNEUMONIA ratio is identical in train and val (both ≈1:2.9). Without stratification, a random split could put disproportionately many NORMAL cases in val, making the effective class distribution different and metrics non-comparable between splits.

**Q: Why convert to grayscale?**
A: X-rays are inherently single-channel intensity images. Any color in the JPEG comes from scanner-specific tinting or compression artifacts — not diagnostic information. Grayscale reduces feature dimensionality 3× at zero information cost.

**Q: Why normalize by mean and std from the training set only?**
A: Using val/test statistics for normalization would be data leakage — the model would implicitly have seen those sets before evaluation. We compute mean/std from 500 random training images and apply the same fixed values to all splits.

**Q: Why is `shuffle=True` for the train loader but `False` for val/test?**
A: Training shuffles each epoch so the model doesn't memorize example order or batch composition (which could constitute implicit regularization bias). Validation and test must be deterministic: the same images, same order, reproducible accuracy numbers.

**Q: Why `num_workers=0`?**
A: On Windows, PyTorch multiprocessing for data loading uses `spawn` to start worker processes. These workers try to inherit file handle state in a way that breaks on Windows. Setting `num_workers=0` loads data in the main process — slower but reliable.

**Q: What augmentations do you apply, and why are they label-preserving?**
A: Horizontal flips, ±10° rotations, random-crop zoom (85-100%). A chest X-ray that is slightly flipped, rotated, or zoomed still shows the same lung pathology. Bacterial consolidation and viral infiltrates are not orientation-specific, so these transforms don't change the label.

---

### SVM

**Q: Explain the SVM hinge loss intuitively.**
A: For each image, compute a score per class. The hinge loss says: "the correct class must beat every other class by at least a margin Δ=1. If any wrong class comes within Δ of the correct class, we pay a penalty proportional to the violation." Classes already well below the correct class contribute zero loss — we don't care by how much we win, only that we win by enough.

**Q: Walk me through the vectorized gradient computation.**
A: We build an indicator matrix (N×C). For each example i, every offending class j gets +1 (it contributes +x_i to dW[:,j]). The correct class y_i gets -|S_i| (the negative count of offenders — it contributes -|S_i|*x_i to dW[:,y_i]). Then `dW = X.T @ indicator / N` is a single matrix multiplication that sums up all contributions across examples simultaneously.

**Q: Why is small weight initialization (0.001) important for the SVM?**
A: We want initial scores to be near zero so all classes are approximately equal at the start. If weights are large, some classes immediately produce huge scores, creating enormous gradients in the first few iterations that destabilize training.

**Q: Why is the numerical gradient check done with central difference rather than forward difference?**
A: Central difference error is O(h²) — much smaller than forward difference's O(h). With h=1e-5, central difference gives ~1e-10 error vs ~1e-5 for forward difference. This makes the check sensitive enough to detect gradient bugs.

**Q: The SVM gets 99.2% PNEUMONIA recall. Does that mean it works well?**
A: No — it's nearly a degenerate classifier. With no class weighting, the SVM learns that predicting PNEUMONIA is almost always "safe" given the 1:2.9 imbalance. NORMAL recall is only 34.2%, meaning it misclassifies 66% of healthy patients as sick. A model that always predicts PNEUMONIA would get 100% PNEUMONIA recall and 0% NORMAL recall.

---

### Convolutional Networks

**Q: Why does a conv layer have far fewer parameters than a fully-connected layer for the same input/output size?**
A: Weight sharing. A 3×3 conv with 32 filters has 32×(3×3+1)=320 parameters regardless of input resolution. An FC layer from 128×128 input to 32×128×128 outputs would need 16384×524288 ≈ 8.6 billion parameters. Weight sharing works because the same patterns (edges, textures) appear at different locations in an image.

**Q: Why padding=1 for a 3×3 convolution?**
A: Without padding, a 3×3 kernel reduces a H×W feature map to (H-2)×(W-2) — the border is lost, and stacking multiple layers would rapidly shrink the spatial size. With padding=1, output size = H + 2×1 - 3 + 1 = H. Spatial dimensions are preserved through the conv; only maxpool reduces them.

**Q: Explain batch normalization and why train/eval mode matters.**
A: BN normalizes activations per channel: subtract mini-batch mean, divide by mini-batch std, then apply learnable scale (γ) and shift (β). During training, it uses statistics computed on the current batch. It simultaneously maintains a running exponential moving average of mean and std. During eval, the fixed running statistics are used — not batch statistics. This gives stable, deterministic inference even for a single image. Forgetting `model.eval()` means BN uses noisy batch stats at test time, which degrades accuracy.

**Q: What does `optimizer.zero_grad()` do and why is it called every iteration?**
A: PyTorch accumulates (adds) gradients into `.grad` on each `loss.backward()` call. Without zeroing, each iteration's gradients pile on top of previous ones, producing incorrect weight updates. `zero_grad()` sets all `.grad` to zero before computing the new gradient.

**Q: Why is Deep CNN accuracy lower than Two-layer CNN despite being a deeper model?**
A: Because of the class-weighted loss, not model capacity. The 1.94× weight on NORMAL errors shifts the decision boundary so the model predicts NORMAL more conservatively. NORMAL recall jumps from 67.9% to 94.4%, but PNEUMONIA recall drops from 97.9% to 75.1%. Since the test set has more PNEUMONIA examples (390 vs 234), missing 25% of them more than offsets the NORMAL improvement in overall accuracy. The deep CNN is not worse — it's optimizing a different objective.

**Q: What is adaptive average pooling and why did you use it instead of a fixed flatten?**
A: `AdaptiveAvgPool2d(4,4)` maps any spatial feature map to a 4×4 grid by averaging. The FC head requires a fixed input size, but spatial resolution after 3 MaxPool blocks depends on input image size (128px → 16px, 224px → 28px). Adaptive pooling decouples the two: the FC layer always sees 128×4×4=2048 features regardless of input resolution. This lets us scale from 128px local dev to 224px on Colab without changing any layer sizes.

**Q: What are logits?**
A: Raw, unnormalized scores output by the final linear layer before any activation. They can be any real number. `argmax(logits)` gives the predicted class. PyTorch's `CrossEntropyLoss` applies softmax internally — never apply softmax before passing to `CrossEntropyLoss` or the probabilities get normalized twice.

**Q: Why `@torch.no_grad()` during inference?**
A: During inference we don't call `loss.backward()`, so we don't need the computation graph. `@torch.no_grad()` disables autograd's graph-building: no intermediate activations are saved for backprop. This reduces memory usage by ~50% and speeds up the forward pass, which matters when evaluating over the full training set.

**Q: Why dropout only in the classifier head and not in the conv blocks?**
A: Conv blocks use BN as their regularizer. Applying dropout to conv feature maps is less effective and can actually hurt by disrupting spatial structure. Dropout is most effective in wide FC layers where co-adaptation between individual neurons is the main overfitting mechanism.

**Q: What is weight_decay in Adam, and how is it different from the class-weighted loss?**
A: Weight decay adds L2 regularization directly on parameter values: `W ← W - lr*(gradient + λ*W)`. It prevents weights from growing too large. The class-weighted loss modifies how much each *training example* contributes to the gradient based on its class. They operate at completely different levels — weight decay regularizes the model; class weighting addresses class imbalance.

---

### Results & Analysis

**Q: The confusion matrix shows high SVM precision for NORMAL (0.964) but low recall (0.342). How is this possible?**
A: Precision = TP/(TP+FP). When the SVM rarely predicts NORMAL, the few times it does are almost always correct (high precision). But recall = TP/(TP+FN) — there are many actual NORMAL cases the SVM never predicts, so FN is huge and recall is low. High precision + low recall means the model is very selective/conservative about predicting that class.

**Q: What does F1 score capture that accuracy misses?**
A: F1 = 2PR/(P+R). On imbalanced data, a model can achieve high accuracy by always predicting the majority class (74.8% for our test set). F1 penalizes this because it equally weights precision and recall. A model with PNEUMONIA recall=0 gets PNEUMONIA F1=0 regardless of precision.

**Q: What is the most dangerous type of error in this task, and which model minimizes it?**
A: Missing a pneumonia case (False Negative for PNEUMONIA = `matrix[1,0]`). A sick patient gets no treatment. The Two-layer CNN minimizes this with 97.9% PNEUMONIA recall. The Deep CNN with class weighting increases NORMAL recall (94.4%) but at the cost of more missed pneumonia cases (75.1% recall) — only appropriate for secondary filtering contexts.

**Q: Why do the CNN validation curves oscillate so much epoch to epoch?**
A: Only 782 validation images. A single poorly-predicted batch represents ~8% of the val set. Random variation in which images appear in which batch, plus the stochastic effects of BN, produce large epoch-to-epoch swings. A larger val set or k-fold cross-validation would produce smoother, more informative curves.

**Q: If you had more time, what would you change?**
A: Transfer learning — fine-tune a pretrained ResNet18 backbone. ImageNet features (edges, textures, shapes) transfer well to X-ray classification. We could freeze all layers except the final FC and achieve better results with far less training. GradCAM attribution maps would also reveal which lung regions the network focuses on, making predictions clinically interpretable.

---

*Good luck with the oral exam.*
