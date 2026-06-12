# Study Guide — Pneumonia X-Ray Classification
### Deep Learning Final Project (99006) — Oral Exam Preparation

---

## Table of Contents
1. [Dataset & Preprocessing](#1-dataset--preprocessing)
2. [Linear SVM — Theory](#2-linear-svm--theory)
3. [Linear SVM — Code](#3-linear-svm--code)
4. [Convolutional Networks — Theory](#4-convolutional-networks--theory)
5. [Two-Layer CNN — Architecture & Code](#5-two-layer-cnn--architecture--code)
6. [Deep CNN — Architecture & Code](#6-deep-cnn--architecture--code)
7. [Training Loop & Optimization](#7-training-loop--optimization)
8. [Results & Analysis](#8-results--analysis)
9. [Likely Oral Exam Questions](#9-likely-oral-exam-questions)

---

## 1. Dataset & Preprocessing

### The dataset
- **Source:** Kermany et al. (Cell, 2018), Guangzhou Women and Children's Medical Center.
- **Size:** 5,863 JPEG chest X-ray images, 2 classes: `NORMAL` (0) and `PNEUMONIA` (1).
- **Patients:** pediatric, age 1–5. All images are anterior-posterior (front-facing).
- **Label quality:** graded by two expert physicians; evaluation set reviewed by a third.

### Folder structure
```
chest_xray/
  train/   NORMAL/ (1341)   PNEUMONIA/ (3875)
  val/     NORMAL/ (8)      PNEUMONIA/ (8)
  test/    NORMAL/ (234)    PNEUMONIA/ (390)
```

### Class imbalance
- Train split: **1:2.9 ratio** (NORMAL:PNEUMONIA). Nearly three times as many pneumonia scans.
- This is common in medical datasets — disease cases are over-represented because they're collected from hospital records.
- If ignored, a classifier can achieve ~74% test accuracy by **always predicting PNEUMONIA**. This looks good numerically but is clinically useless.

### Why we re-split the validation set
- The official val folder has only **16 images (8 per class)** — far too small to tune hyperparameters reliably.
- One misclassified image changes val accuracy by 6.25 percentage points. Noise dominates signal.
- **Solution:** carve a stratified 15% hold-out from training data (`make_stratified_val_split`).
  - Stratified = same class ratio in both halves. Implemented by shuffling each class independently and taking the first 15%.
  - Result: 782 val images (201 NORMAL / 581 PNEUMONIA), preserving the 1:2.9 ratio.
- The test set (624 images) is **never touched** during training or tuning.

### Quality-control screening (`screen_images`)
Mirrors the paper's description: "all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans."
- Opens each file with PIL's `verify()` (checks file integrity without full decoding), then re-opens to read size.
- Drops: (a) corrupt/unreadable files, (b) images smaller than 32×32 px.
- In practice, 0 files were removed — the published dataset is already clean.

### Image preprocessing pipeline

**For the SVM** (`load_images_as_arrays`):
1. Convert to grayscale (`img.convert("L")`) — removes color tinting artifacts from different scanners.
2. Resize to 64×64 (`PIL.Image.resize`) — yields a feature vector of 4096 dimensions.
3. Flatten and divide by 255 — puts pixels in [0, 1].
4. After loading all splits, subtract the **training mean** (computed only from train) to center the data.

> **Why subtract the mean?** Centering makes optimization easier because it keeps gradients balanced around zero. You must use training statistics only — applying test statistics would be data leakage.

**For the CNNs** (`build_transform`, `compute_mean_std`):
1. Grayscale conversion.
2. Resize to 128×128 (larger than SVM to preserve spatial structure for convolution).
3. Convert to tensor (PIL → float32 in [0, 1] via `transforms.ToTensor()`).
4. Normalize: subtract training mean (≈0.485) and divide by training std (≈0.236).
   - Mean/std estimated from a random sample of 500 training images.
   - Normalization ensures activations in early layers start in a reasonable range.
5. **Augmentation (train only):** horizontal flips (50%), ±10° rotation, random-crop zoom (85–100%).
   - These are **label-preserving** for chest X-rays: a flipped or slightly rotated lung still has the same diagnosis.
   - Augmentation artificially enlarges the training set and reduces overfitting.

### Class weights (`compute_class_weights`)
$$w_c = \frac{N}{C \cdot n_c}$$
where $N$ = total training samples, $C$ = number of classes, $n_c$ = samples in class $c$.
- NORMAL weight ≈ 1.94, PNEUMONIA weight ≈ 0.67.
- Passed to `nn.CrossEntropyLoss(weight=...)`: losses on NORMAL misclassifications are multiplied by 1.94, making the network pay more attention to the minority class.

---

## 2. Linear SVM — Theory

### What is the multiclass SVM?
A **linear classifier** of the form $s = Wx$ where $W \in \mathbb{R}^{D \times C}$ is the weight matrix, $x \in \mathbb{R}^D$ is the flattened image, and $s \in \mathbb{R}^C$ are the class scores.

The SVM is trained to produce scores such that the correct class score **exceeds every other class score by at least a margin $\Delta$**.

### The hinge loss (Weston-Watkins formulation)

$$L = \frac{1}{N}\sum_{i=1}^{N} \sum_{j \ne y_i} \max\!\left(0,\; s_j - s_{y_i} + \Delta\right) + R(W)$$

- $y_i$ = correct class label for example $i$.
- $s_{y_i}$ = score for the correct class.
- $\Delta = 1$ = the required safety margin.
- The **hinge** ($\max(0, \cdot)$) means we only penalize a class $j$ if its score is within $\Delta$ of the correct class — we don't care *how* much the correct class wins, as long as it wins by at least $\Delta$.
- $R(W)$ = regularization term (see below).

### Why $\Delta = 1$?
The value of $\Delta$ is arbitrary because the regularization strength $\lambda$ can compensate. Setting $\Delta = 1$ is a convention — what matters is the ratio $\Delta / \lambda$.

### Gradient of the hinge loss
For each example $i$, define the set of "offending classes" $S_i = \{j \ne y_i : s_j - s_{y_i} + \Delta > 0\}$.

$$\frac{\partial L_i}{\partial W_j} = \begin{cases} x_i & \text{if } j \in S_i \\ -|S_i| \cdot x_i & \text{if } j = y_i \\ 0 & \text{otherwise} \end{cases}$$

**Intuition:** every offending class $j$ "pulls" the weight column $W_j$ toward $x_i$ (because it scored too high). The correct class column is pushed down once per offending class. Average over the batch, add regularization gradient.

### L2 vs L1 Regularization

| Property | L2 ($\lambda \|W\|_F^2$) | L1 ($\lambda \|W\|_1$) |
|----------|--------------------------|------------------------|
| Gradient | $2\lambda W$ | $\lambda \cdot \text{sign}(W)$ |
| Effect on weights | Shrinks all weights smoothly toward 0 | Drives many weights exactly to 0 (sparsity) |
| Geometry | Penalizes large weights; encourages small, spread-out weights | Encourages sparse weight vectors |
| Differentiability | Smooth everywhere | Not differentiable at 0 (use subgradient: sign) |

**Why L2 is usually preferred for SVMs:** L2 gives a unique, smooth solution and corresponds to the classical SVM margin maximization. L1 is useful when feature sparsity is desired. In our results, both performed nearly identically (74.8% vs 74.5%) — the pixel feature representation is the bottleneck, not regularization.

### The bias trick
Instead of maintaining a separate bias vector $b$, we append a constant 1 to every input:
$$\tilde{x} = [x_1, x_2, \ldots, x_D, 1] \in \mathbb{R}^{D+1}$$
Then $W\tilde{x}$ incorporates the bias as the last row of $W$. This simplifies the code — we optimize one matrix instead of a matrix and a vector.

### Numerical gradient check
The central difference formula approximates $\frac{\partial L}{\partial W_{ij}}$:
$$\frac{\partial L}{\partial W_{ij}} \approx \frac{L(W_{ij} + h) - L(W_{ij} - h)}{2h}$$
with $h = 10^{-5}$. We compare this to the analytic gradient at random coordinates. If the relative error $\frac{|g_\text{analytic} - g_\text{numeric}|}{|g_\text{analytic}| + |g_\text{numeric}|}$ is below $10^{-9}$, the gradient is correct. Our check passed with error ≈ $7 \times 10^{-11}$.

### Mini-batch SGD
Instead of computing the gradient over the entire training set (expensive), we sample a random batch of 200 examples per iteration and compute the gradient on that batch. This gives a **noisy but cheap estimate** of the true gradient.
- 1500 iterations × batch 200 = sees each example ~67 times on average.
- Weight update: $W \leftarrow W - \eta \cdot \nabla_W L_\text{batch}$

---

## 3. Linear SVM — Code

### `svm_loss_naive` (linear_svm.py:15)
```python
for i in range(num_train):
    scores = X[i].dot(W)              # shape (C,)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
        if j == y[i]: continue
        margin = scores[j] - correct_class_score + delta
        if margin > 0:
            loss += margin
            dW[:, j] += X[i]          # offending class column
            dW[:, y[i]] -= X[i]       # correct class column loses once per offender
```
Explicit loops — slow but each line maps directly to the math.

### `svm_loss_vectorized` (linear_svm.py:55)
```python
scores = X.dot(W)                                          # (N, C)
correct_class_scores = scores[np.arange(N), y].reshape(-1,1)  # (N, 1)
margins = np.maximum(0.0, scores - correct_class_scores + delta)  # (N, C)
margins[np.arange(N), y] = 0.0                             # zero out correct class
loss = margins.sum() / N

indicator = (margins > 0).astype(W.dtype)                  # (N, C)
indicator[np.arange(N), y] = -indicator.sum(axis=1)        # correct class = -|S_i|
dW = X.T.dot(indicator) / N                               # (D, C)
```
Key insight: `indicator[i, j]` is 1 for offending classes and $-|S_i|$ for the correct class — exactly the gradient formula vectorized over all examples.

### `LinearClassifier.train` (linear_svm.py:130)
1. Appends bias column of ones to X.
2. Initialises W with small random values (`0.001 * randn`) — near-zero scores at start.
3. Each iteration: sample batch → compute loss+gradient → `W -= lr * dW`.

### Hyperparameter sweep result
Best L2: lr=0.1, reg=0.001. Grid searched over 3 learning rates × 3 reg strengths on val set.

---

## 4. Convolutional Networks — Theory

### What is a convolution?
A 2D convolution slides a small **filter** (kernel) of shape $k \times k$ across the image, computing a dot product at each position. For a filter of size 3×3 with padding=1:
- Output spatial size = input size (padding preserves dimensions).
- Each filter learns to detect a specific local pattern (edge, texture, blob).
- With $F$ filters: output has $F$ channels, each a "feature map."
- **Weight sharing:** the same filter weights are used at every spatial location → far fewer parameters than a fully-connected layer.

### Why ReLU?
$$\text{ReLU}(x) = \max(0, x)$$
- Introduces non-linearity (without which stacked linear layers collapse to one linear layer).
- Computationally cheap (just a threshold).
- Does not saturate for positive inputs (unlike sigmoid/tanh), so gradients flow better.
- Sparsity: many units output 0, which can be beneficial for representation.

### What does MaxPool do?
`MaxPool2d(2)` divides the feature map into non-overlapping 2×2 tiles and keeps the maximum value in each tile.
- **Spatial resolution halves** (128→64→32→16 with 3 blocks).
- Provides **translation invariance**: a feature detected slightly off-center still activates the pooled output.
- Reduces the number of parameters in subsequent layers.

### Batch Normalization
Applied after each convolution, before ReLU, in the deep network:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- $\mu_B, \sigma_B^2$ = mean and variance over the current mini-batch (per channel).
- $\gamma, \beta$ = learnable scale and shift parameters.
- **Why it helps:**
  1. Keeps activation distributions stable across layers → allows higher learning rates.
  2. Acts as implicit regularization (adds noise from batch statistics).
  3. Reduces sensitivity to weight initialization.
  4. Critical here because X-rays from different scanners have different contrast/brightness — BN normalizes these out at each layer.
- **Train vs eval mode:** during training uses batch statistics; during evaluation uses running mean/std accumulated over training. PyTorch handles this automatically with `model.train()` / `model.eval()`.

### Dropout
`Dropout(p=0.5)` randomly sets 50% of neurons to zero during each forward pass in training.
- Forces the network to learn redundant representations — no single neuron can be relied upon.
- Equivalent to training an ensemble of $2^n$ sub-networks and averaging them at test time.
- Applied only in the **classifier head** (after the convolutional features), not in the conv blocks.
- At test time: disabled (`model.eval()`). PyTorch handles this automatically.

### Adaptive Average Pooling
`AdaptiveAvgPool2d((4, 4))` takes the feature map (whatever spatial size it is after 3 max-pool blocks) and outputs a fixed 4×4 grid by averaging.
- **Why useful:** the classifier head (`Linear`) needs a fixed input size. With adaptive pooling, the network works with any input resolution — we can scale up to 224×224 on Colab without changing the FC layer.
- The flat dimension after this layer = `channels[-1] × 4 × 4 = 128 × 16 = 2048`.

### Cross-entropy loss
For a 2-class problem:
$$L = -\frac{1}{N}\sum_{i=1}^{N} \log\!\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right) = -\frac{1}{N}\sum_{i=1}^{N} \log p_{y_i}$$
where $p_j = \text{softmax}(s)_j$ is the predicted probability. The loss is the average negative log-probability assigned to the correct class.

**With class weights:** `CrossEntropyLoss(weight=[w_0, w_1])` multiplies each example's loss by $w_{y_i}$, penalizing errors on NORMAL ($w_0 \approx 1.94$) nearly twice as much as errors on PNEUMONIA ($w_1 \approx 0.67$).

### Adam optimizer
Adam maintains per-parameter moving averages of the gradient ($m$) and squared gradient ($v$):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$W \leftarrow W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
where $\hat{m}, \hat{v}$ are bias-corrected. This gives **adaptive, per-parameter learning rates** — parameters with historically large gradients get smaller effective learning rates. More robust than plain SGD for neural networks.

---

## 5. Two-Layer CNN — Architecture & Code

### Architecture (`conv_nets.py:16`)
```
Input: (batch, 1, 128, 128)
  → Conv2d(1→16, 3×3, pad=1)  →  (batch, 16, 128, 128)
  → ReLU
  → MaxPool2d(2)               →  (batch, 16, 64, 64)
  → Flatten                    →  (batch, 16×64×64) = (batch, 65536)
  → Linear(65536 → 256)
  → ReLU
  → Linear(256 → 2)            →  (batch, 2)  [logits]
```

**Parameter count:**
- Conv: 16 × (1×3×3 + 1 bias) = 160
- FC1: 65536 × 256 + 256 = ~16.8M
- FC2: 256 × 2 + 2 = 514
- **Total: ~16.8M** — most parameters are in FC1.

**`flat_dim` calculation (`conv_nets.py:31`):**
```python
pooled_size = img_size // 2   # 128 // 2 = 64
flat_dim = num_filters * pooled_size * pooled_size  # 16 * 64 * 64 = 65536
```

### Sanity check: overfit a tiny batch
Before full training we verify the network can achieve ~100% accuracy on 32 samples after 40 epochs.
- **Why:** if a network can't overfit a tiny batch, there's a bug in the forward pass, loss, or optimizer setup. Overfitting a tiny batch confirms gradients flow correctly.
- Result: train_acc reached 1.000 — check passed.

### Results
- Test accuracy: **86.7%** (vs SVM 74.8% — gain of +11.9%)
- NORMAL recall: **67.9%** (vs SVM 34.2% — nearly doubled)
- PNEUMONIA recall: **97.9%**
- Training ran for 20 epochs, converged well.

---

## 6. Deep CNN — Architecture & Code

### Architecture (`conv_nets.py:44`)
```
Input: (batch, 1, 128, 128)
  ┌─ Block 1: Conv2d(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
  │           (batch, 32, 64, 64)
  ├─ Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
  │           (batch, 64, 32, 32)
  └─ Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
              (batch, 128, 16, 16)
  → AdaptiveAvgPool2d(4×4)      (batch, 128, 4, 4)
  → Flatten                     (batch, 2048)
  → Linear(2048 → 256) → ReLU
  → Dropout(0.5)
  → Linear(256 → 2)             (batch, 2)
```

**Why channels double at each block (32→64→128)?**
As spatial resolution halves, we compensate by doubling channels. This keeps the total information capacity roughly constant across depth levels and is a widely-used empirical design principle.

**Parameter count (approximate):**
- Block 1: 32×(1×3×3 +1) = 320
- Block 2: 64×(32×3×3+1) ≈ 18.5K
- Block 3: 128×(64×3×3+1) ≈ 73.9K
- FC1: 2048×256 ≈ 524K
- FC2: 256×2 = 514
- BN parameters: small (2 per channel per block)
- **Total: ~617K** — much smaller than TwoLayerConvNet's 16.8M because adaptive pooling replaced a huge FC layer.

### How the blocks are built programmatically
```python
blocks = []
prev_channels = in_channels   # starts at 1
for out_channels in channels:  # (32, 64, 128)
    blocks.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1))
    blocks.append(nn.BatchNorm2d(out_channels))
    blocks.append(nn.ReLU())
    blocks.append(nn.MaxPool2d(kernel_size=2))
    prev_channels = out_channels
self.features = nn.Sequential(*blocks)
```
A simple loop appends layers; `nn.Sequential(*blocks)` chains them.

### Class-weighted loss
```python
class_weights = du.compute_class_weights(train_items)  # [1.944, 0.673]
weighted_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights).to(device)
)
```
The formula is: `w_c = N / (C * n_c)`. For our data: NORMAL has 1140/4434 ≈ 25.7% of samples, so its weight = 1/(2×0.257) ≈ 1.944.

---

## 7. Training Loop & Optimization

### The training loop (`solver.py:38`)
```python
for epoch in range(num_epochs):
    model.train()                      # enables dropout and BN train mode
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()          # clear gradients from previous step
        logits = model(images)         # forward pass
        loss = criterion(logits, targets)
        loss.backward()                # compute gradients via backprop
        optimizer.step()               # update weights
        history["train_loss"].append(float(loss.item()))

    model.eval()                       # disable dropout; use BN running stats
    train_acc = evaluate_accuracy(model, train_loader, device)
    val_acc   = evaluate_accuracy(model, val_loader, device)
```

**`optimizer.zero_grad()`** — PyTorch accumulates gradients by default. Must be called before each backward pass to reset them.

**`loss.backward()`** — computes $\frac{\partial \text{loss}}{\partial \theta}$ for all parameters $\theta$ via automatic differentiation (reverse-mode autodiff / backpropagation).

**`model.train()` / `model.eval()`** — critical switch. In eval mode: Dropout is disabled (all neurons active); BatchNorm uses running statistics instead of batch statistics. Always call `model.eval()` before validation/test evaluation.

**`@torch.no_grad()`** on `predict_all` — disables gradient computation during inference, saving memory and compute (no need to build the computation graph).

### Adam hyperparameters used
- Learning rate: `1e-3`
- Weight decay (L2 on weights): `1e-4`
- Default β₁=0.9, β₂=0.999

---

## 8. Results & Analysis

### Summary table

| Model | Test Acc | NORMAL Recall | PNEUMONIA Recall |
|-------|:--------:|:-------------:|:----------------:|
| Linear SVM (L2) | 74.8% | 34.2% | 99.2% |
| Linear SVM (L1) | 74.5% | 33.8% | 99.0% |
| Two-layer CNN | **86.7%** | 67.9% | **97.9%** |
| Deep CNN | 82.4% | **94.4%** | 75.1% |

### Why is SVM test accuracy only 74.8% despite val accuracy 95.9%?
Two reasons:
1. **Class imbalance in the val set.** The re-split val set is 74.3% PNEUMONIA. If the SVM nearly always predicts PNEUMONIA, val accuracy ≈ 74.3% — but the SVM reaches 95.9%, which suggests the small val set is overfit by the hyperparameter sweep.
2. **Distribution shift.** The test set has 37.5% NORMAL (vs 25.7% in val). An SVM biased toward PNEUMONIA will perform worse on a test set with proportionally more NORMAL cases. This is why the gap is so large.

### Why does the SVM have 99.2% PNEUMONIA recall but only 34.2% NORMAL recall?
With a 1:2.9 imbalance and no class weighting, the SVM learns that predicting PNEUMONIA is almost always "safe." It becomes a **near-degenerate classifier** that rarely predicts NORMAL. The high PNEUMONIA recall is a consequence of this bias, not genuine discrimination.

A model that always predicts PNEUMONIA would get: PNEUMONIA recall=100%, NORMAL recall=0%, test acc=390/624=62.5%. Our SVM is slightly better than this but the pattern is the same.

### Why does L1 vs L2 barely matter for the SVM?
Both achieve the same test accuracy (74.8% vs 74.5%). The bottleneck is not overfitting — it's that the pixel feature representation cannot linearly separate pneumonia from normal at this resolution. No amount of regularization can fix a fundamentally insufficient feature space.

### Why does the Deep CNN have LOWER accuracy than the Two-layer CNN?
This is the most important result to understand. The Deep CNN uses **class-weighted loss** with NORMAL weight ≈ 1.94.

This shifts the decision boundary: the network is penalized more for misclassifying NORMAL, so it becomes more conservative about predicting PNEUMONIA. The result:
- NORMAL recall jumps from 67.9% to **94.4%** — catches nearly all healthy patients.
- PNEUMONIA recall drops from 97.9% to **75.1%** — misses more pneumonia cases.
- Overall accuracy drops because there are more PNEUMONIA cases in the test set, and we now misclassify more of them.

This is not a failure — it's a **deliberate, clinically motivated trade-off** controlled by the loss weights.

### Precision vs Recall — clinical interpretation
- **Precision** = of all cases I predicted PNEUMONIA, how many truly have it.
- **Recall** = of all true pneumonia cases, how many did I catch.

For a **screening tool** (first-line triage): maximize PNEUMONIA recall — missing a sick patient is dangerous. → Two-layer CNN (97.9% PNEUMONIA recall) is better.

For a **secondary filter** (reducing unnecessary antibiotics): maximize NORMAL recall — flagging a healthy patient for unnecessary treatment is costly. → Deep CNN (94.4% NORMAL recall) is better.

### Reading the confusion matrix
The matrix has **rows = true label, columns = predicted label**:
```
              Predicted
              NORMAL   PNEUMONIA
True NORMAL  [  TP_N     FN_N  ]
True PNEU    [  FP_N     TP_P  ]
```
- Top-left: correctly identified normal patients.
- Bottom-right: correctly identified pneumonia patients.
- Top-right (False Negatives for NORMAL): healthy patients flagged as sick.
- Bottom-left (False Negatives for PNEUMONIA): sick patients missed — the most dangerous error.

For the SVM: the top-right cell is very large (misses ~66% of healthy patients).

---

## 9. Likely Oral Exam Questions

### Dataset & Preprocessing
**Q: Why did you re-split the validation set?**
A: The official val folder has 16 images — 8 per class. One wrong prediction is a 6.25% swing in accuracy. It's statistically meaningless for tuning. We carved a stratified 15% hold-out from training (782 images) so tuning decisions are reliable.

**Q: What does stratified mean, and why does it matter?**
A: Stratified means we hold out 15% from *each class separately*, preserving the 1:2.9 ratio in both train and val. Without stratification, a random split could accidentally put more NORMAL cases in val, changing the effective distribution and making metrics non-comparable.

**Q: Why convert to grayscale?**
A: X-rays are inherently single-channel intensity images. Color channels in JPEG files represent scanner-specific tinting or compression artifacts, not diagnostic information. Grayscale reduces input dimensionality by 3× at no information cost.

**Q: Why mean/std normalize the CNN input?**
A: Neural networks are sensitive to input scale. Without normalization, pixels in [0,255] would produce very large activations in early layers, destabilizing training. Subtracting mean and dividing by std puts inputs near zero with unit variance, matching the regime where gradient magnitudes are balanced.

**Q: Why use only training statistics for normalization?**
A: Using test statistics would be data leakage — the model would have implicitly "seen" the test set before evaluation. We fix the normalization parameters from training data only.

---

### SVM
**Q: Explain the SVM hinge loss intuitively.**
A: For each training image, we compute a score for every class. The hinge loss says: "the correct class score must beat every other class score by at least margin Δ=1. If any wrong class comes within Δ of the correct class score, we pay a penalty proportional to how close it is." Classes that are far below the margin contribute zero loss.

**Q: What is the gradient of the SVM loss?**
A: For each offending wrong class $j$ the gradient of the loss w.r.t. column $W_j$ is $+x_i$. Gradient descent then does $W_j \mathrel{-}= \eta \cdot x_i$, pulling that column *away* from $x_i$ and lowering the wrong-class score $W_j \cdot x_i$. The correct class column gets gradient $-|S_i| \cdot x_i$; the descent step adds $\eta |S_i| x_i$, pushing $W_{y_i}$ *toward* $x_i$ and raising the correct score. Averaged over the batch, plus the regularization term.

**Q: What is the bias trick?**
A: Appending a constant 1 to every input vector absorbs the bias into the weight matrix. Instead of computing $Wx + b$, we compute $\tilde{W}\tilde{x}$ where $\tilde{x} = [x; 1]$ and the last row of $\tilde{W}$ acts as the bias. This simplifies code with no mathematical change.

**Q: Why is the numerical gradient check important?**
A: The vectorized gradient derivation is error-prone. A numerical check using the central difference formula independently verifies the analytic gradient is correct. If they disagree, there's a bug. We check at random weight coordinates and verify relative error < $10^{-9}$.

**Q: Why does regularization not help the SVM here?**
A: The bottleneck is the feature representation (4096 raw pixel values) which has insufficient structure to linearly separate the two classes. Regularization prevents overfitting, but there's nothing to overfit when the model can't fit in the first place. L1 and L2 give identical test accuracy (74.8% vs 74.5%).

---

### CNNs
**Q: Why does a convolutional layer have far fewer parameters than a fully-connected layer?**
A: Weight sharing. A 3×3 conv with 32 filters has 32×(3×3×1+1)=320 parameters, regardless of input image size. An equivalent FC layer mapping 128×128=16384 inputs to the same 32×128×128 outputs would have 16384×524288 ≈ 8.6 billion parameters.

**Q: What does max-pooling do?**
A: Divides the feature map into non-overlapping tiles (2×2 here) and keeps the maximum activation per tile. This halves spatial dimensions, reducing computation and providing local translation invariance — a feature detected slightly off-center still activates the pooled output.

**Q: Explain batch normalization and why it's especially useful here.**
A: BN normalizes activations across the batch after each conv layer: subtract batch mean, divide by batch std, then apply learnable scale (γ) and shift (β). This keeps the distribution of activations stable layer to layer, allowing higher learning rates and acting as regularization. It's especially useful for X-rays because images from different scanners have different brightness/contrast — BN normalizes these systematic differences at each layer.

**Q: Why does batch normalization behave differently at train vs test time?**
A: During training, normalization uses the *current mini-batch* statistics (mean/std computed on that batch). During inference, using a single image's statistics would be noisy. Instead, BN tracks a running exponential moving average of mean and std during training, and uses these fixed values at test time. PyTorch switches modes via `model.train()` / `model.eval()`.

**Q: What is dropout and how does it prevent overfitting?**
A: Dropout randomly zeroes a fraction $p$ of neuron outputs during each forward pass. This prevents neurons from co-adapting — no single neuron can rely on specific other neurons always being present. It's equivalent to averaging predictions over an exponential number of sub-networks. At test time, dropout is disabled and all neurons are active (with outputs scaled by $1-p$ during training to keep expected values consistent).

**Q: Why is the deep CNN's accuracy lower than the two-layer CNN's?**
A: Because of the class-weighted loss. Weighting NORMAL errors 1.94× more causes the deep CNN to shift its decision boundary: it becomes more conservative about predicting PNEUMONIA, correctly identifying 94.4% of healthy patients but missing 25% of pneumonia cases. Since pneumonia cases are more common in the test set, missing more of them reduces overall accuracy. This is a deliberate trade-off, not a failure.

**Q: What is adaptive average pooling and why is it useful?**
A: `AdaptiveAvgPool2d(4,4)` maps any spatial feature map to a fixed 4×4 grid by averaging within a variable-size window. It decouples the spatial size (which changes with input resolution) from the classifier head (which needs a fixed input). This means we can increase image resolution from 128 to 224 on Colab without changing any FC layer dimensions.

**Q: Why do channels double at each conv block?**
A: Each max-pool halves the spatial resolution (halving information capacity per channel). Doubling channels compensates, maintaining roughly constant total representation capacity throughout the network. This is an empirical design principle borrowed from VGG/ResNet-family architectures.

---

### Training & Results
**Q: What does `optimizer.zero_grad()` do and why is it necessary?**
A: PyTorch accumulates gradients by default — `loss.backward()` *adds* to existing `.grad` attributes. Without zeroing, gradients from multiple batches would be summed, producing incorrect weight updates. Must be called before each `backward()`.

**Q: Why do we call `model.eval()` before evaluation?**
A: Switches dropout (off) and batch normalization (uses running stats, not batch stats) to their inference modes. Without this, evaluation would be stochastic (different results each run) and batch norm would use noisy single-batch statistics instead of stable training-set estimates.

**Q: What does the loss curve tell you?**
A: A decreasing, smooth loss curve means training is progressing. Oscillation indicates a high learning rate or small batch size. If training loss decreases but val accuracy stagnates or drops, the model is overfitting.

**Q: The SVM val accuracy is 95.9% but test accuracy is only 74.8%. What went wrong?**
A: Two factors. First, the re-split val set is 74.3% PNEUMONIA (581/782) while the original test set is only 62.5% PNEUMONIA (390/624). An SVM biased toward predicting PNEUMONIA scores higher on the PNEUMONIA-heavy val set. Second, the small val set (782 images) makes the hyperparameter sweep overfit — the best configuration by chance looks good on val but doesn't generalise. This gap (95.9% → 74.8%) illustrates why a representative, sufficiently large validation set is critical.

**Q: Explain precision and recall in this context.**
A: For PNEUMONIA:
- **Precision** = of all patients the model flagged as sick, what fraction were actually sick. High precision = few false alarms.
- **Recall** = of all actually sick patients, what fraction did the model catch. High recall = few missed diagnoses.
In screening, recall is paramount — missing a pneumonia case means an untreated patient. In a follow-up setting, precision matters more — unnecessary treatment is costly.

**Q: What is the F1 score?**
A: The harmonic mean of precision and recall: $F1 = 2 \cdot \frac{P \cdot R}{P + R}$. It summarizes both in a single number, penalizing heavily when either is low. More informative than accuracy on imbalanced datasets.

---

*Good luck with the oral exam.*
