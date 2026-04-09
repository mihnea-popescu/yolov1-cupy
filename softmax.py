import cupy as cp


def softmax(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    """
    Turn unnormalized scores into probabilities (non-negative, sum to 1 along ``axis``).

    Subtracts the per-row max for numerical stability before exp.
    """
    x_max = x.max(axis=axis, keepdims=True)
    ex = cp.exp(x - x_max)
    return ex / ex.sum(axis=axis, keepdims=True)
