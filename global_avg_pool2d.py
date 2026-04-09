import cupy as cp


class GlobalAvgPool2D:
    """
    Global average pooling over spatial dimensions (H, W).

    Forward: (N, C, H, W) -> (N, C). No learnable parameters.
    """

    def __init__(self):
        self._input_shape = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        assert x.ndim == 4, f"Expected (N, C, H, W), got {x.shape}"
        self._input_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        assert self._input_shape is not None, "Call forward() before backward()"
        n, c, h, w = self._input_shape
        assert grad_output.shape == (n, c), (
            f"Expected grad ({n}, {c}), got {grad_output.shape}"
        )
        scale = 1.0 / (h * w)
        return grad_output[:, :, None, None] * scale

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return "GlobalAvgPool2D()"
