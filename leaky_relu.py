import cupy as cp


class LeakyReLU:
    def __init__(self, negative_slope: float = 0.1):
        self.negative_slope = negative_slope
        self._mask = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self._mask = (x > 0).astype(x.dtype)
        out = cp.where(x > 0, x, self.negative_slope * x)
        return out

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        assert self._mask is not None, "forward() has not been called"
        dx = grad_output * cp.where(self._mask, 1.0, self.negative_slope)
        return dx

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"