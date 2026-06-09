"""A linear support vector machine classifier implemented from scratch in NumPy.

This follows the classic multiclass SVM (hinge / max-margin) formulation.  We
build it up incrementally: a slow but transparent ``svm_loss_naive`` with
explicit loops, then a fast ``svm_loss_vectorized`` that must agree with it, and
finally a ``LinearClassifier`` that trains the weights with mini-batch SGD.

Both L2 and L1 weight regularization are supported so we can compare them, as
the assignment asks.
"""

import numpy as np


def svm_loss_naive(W, X, y, reg, reg_type="l2", delta=1.0):
    """Multiclass SVM loss and gradient, computed with explicit loops.

    Args:
        W: weight matrix of shape (D, C).
        X: data matrix of shape (N, D).
        y: integer labels of shape (N,), each in [0, C).
        reg: regularization strength.
        reg_type: 'l2' or 'l1'.
        delta: the margin required between the correct and other scores.

    Returns (loss, gradient w.r.t. W).
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                # This class pushed past the margin, so it gets a gradient.
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Average the data loss and gradient over the batch.
    loss /= num_train
    dW /= num_train

    # Add the regularization penalty on the weights.
    loss, dW = _add_regularization(loss, dW, W, reg, reg_type)
    return loss, dW


def svm_loss_vectorized(W, X, y, reg, reg_type="l2", delta=1.0):
    """Vectorized version of the multiclass SVM loss; matches the naive one."""
    num_train = X.shape[0]

    scores = X.dot(W)
    # Pull out the score of the correct class for every example.
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    margins = np.maximum(0.0, scores - correct_class_scores + delta)
    # The correct class never contributes its own +delta term.
    margins[np.arange(num_train), y] = 0.0
    loss = margins.sum() / num_train

    # Gradient: each class with a positive margin contributes X[i]; the correct
    # class loses X[i] once per offending class.
    indicator = (margins > 0).astype(W.dtype)
    indicator[np.arange(num_train), y] = -indicator.sum(axis=1)
    dW = X.T.dot(indicator) / num_train

    loss, dW = _add_regularization(loss, dW, W, reg, reg_type)
    return loss, dW


def _add_regularization(loss, dW, W, reg, reg_type):
    """Add the L2 or L1 weight penalty to the loss and gradient."""
    if reg_type == "l2":
        loss += reg * np.sum(W * W)
        dW += 2 * reg * W
    elif reg_type == "l1":
        loss += reg * np.sum(np.abs(W))
        dW += reg * np.sign(W)
    else:
        raise ValueError(f"unknown reg_type {reg_type!r}")
    return loss, dW


def numeric_gradient(loss_fn, W, h=1e-5, num_checks=10, seed=0):
    """Central-difference numerical gradient at a few random coordinates.

    Used as a sanity check against the analytic gradient.  ``loss_fn`` must take
    only W and return the scalar loss.  Returns a list of (analytic, numeric,
    relative_error) tuples for the sampled coordinates.
    """
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(num_checks):
        ix = tuple(rng.randint(dim) for dim in W.shape)
        old_value = W[ix]

        W[ix] = old_value + h
        loss_plus = loss_fn(W)
        W[ix] = old_value - h
        loss_minus = loss_fn(W)
        W[ix] = old_value  # restore

        numeric = (loss_plus - loss_minus) / (2 * h)
        results.append((ix, numeric))
    return results


class LinearClassifier:
    """A linear SVM trained with mini-batch stochastic gradient descent.

    A bias term is handled with the usual "bias trick": we append a constant
    column of ones to the inputs, so the bias becomes an extra row of W and we
    do not have to track it separately.
    """

    def __init__(self):
        self.W = None

    @staticmethod
    def _append_bias(X):
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])

    def train(self, X, y, learning_rate=1e-2, reg=1e-3, reg_type="l2",
              num_iters=1000, batch_size=128, seed=0, verbose=False):
        """Run SGD on the SVM loss and return the per-iteration loss history."""
        X = self._append_bias(X)
        num_train, dim = X.shape
        num_classes = int(np.max(y)) + 1

        rng = np.random.RandomState(seed)
        if self.W is None:
            # Small random init keeps the initial scores near zero.
            self.W = 0.001 * rng.randn(dim, num_classes).astype(X.dtype)

        loss_history = []
        for it in range(num_iters):
            batch_idx = rng.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            loss, dW = svm_loss_vectorized(self.W, X_batch, y_batch, reg, reg_type)
            loss_history.append(loss)
            # Gradient descent step.
            self.W -= learning_rate * dW

            if verbose and it % 100 == 0:
                print(f"  iter {it:4d}/{num_iters}  loss {loss:.4f}")
        return loss_history

    def predict(self, X):
        """Return the predicted class label for each row of X."""
        X = self._append_bias(X)
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)
