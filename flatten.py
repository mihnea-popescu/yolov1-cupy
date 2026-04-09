import cupy as cp
import cupyx


class Flatten:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        assert len(self.x_shape) > 1, f"Expects at least 2D input, got {len(self.x_shape)}"
        out = x.reshape(self.x_shape[0], -1)
        return out

    def backward(self, grad_output):
        assert self.x_shape is not None, "forward() has not been called"
        return grad_output.reshape(self.x_shape)
