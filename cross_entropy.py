import cupy as cp

from softmax import softmax


def softmax_cross_entropy_grad(
    logits: cp.ndarray,
    y,
    *,
    mean_over_batch: bool = True,
) -> cp.ndarray:
    """
    Gradient dL/d(logits) for softmax + cross-entropy.

    ``y`` is class indices (N,) — NumPy or CuPy int64.
    If ``mean_over_batch`` is True, loss is mean CE (standard); else sum CE.
    """
    y = cp.asarray(y, dtype=cp.int64)
    n, num_classes = logits.shape
    assert y.shape == (n,), (y.shape, n)

    probs = softmax(logits, axis=1)
    one_hot = cp.zeros_like(probs)
    one_hot[cp.arange(n), y] = 1.0
    grad = probs - one_hot
    if mean_over_batch:
        grad /= n
    return grad


def softmax_cross_entropy_loss(
    logits: cp.ndarray,
    y,
    *,
    mean_over_batch: bool = True,
) -> float:
    """Mean (or sum) softmax cross-entropy; log-sum-exp for stability."""
    y = cp.asarray(y, dtype=cp.int64)
    n, _ = logits.shape
    assert y.shape == (n,), (y.shape, n)
    m = logits.max(axis=1, keepdims=True)
    log_sum_exp = cp.log(cp.exp(logits - m).sum(axis=1, keepdims=True))
    log_probs = (logits - m) - log_sum_exp
    nll = -log_probs[cp.arange(n), y]
    loss = nll.sum()
    if mean_over_batch:
        loss = loss / n
    return float(cp.asnumpy(loss))
