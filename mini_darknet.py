import cupy as cp

from batchnorm2d import BatchNorm2D
from conv2d import Conv2D
from global_avg_pool2d import GlobalAvgPool2D
from leaky_relu import LeakyReLU
from linear import Linear
from maxpool import MaxPool2D


class MiniDarknet:
    """
    Mini Darknet backbone + classifier head.

    Input: (N, 3, 224, 224). Output: (N, num_classes) logits.
    Call ``forward`` before ``backward``. ``grad_logits`` is dL/d(logits),
    same shape as logits. Returns dL/d(input images), shape (N, 3, 224, 224).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dtype=cp.float64,
        negative_slope: float = 0.1,
    ):
        self.dtype = dtype
        channels = [(3, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
        self.blocks = []
        for cin, cout in channels:
            self.blocks.append(
                (
                    Conv2D(
                        cin,
                        cout,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        dtype=dtype,
                    ),
                    BatchNorm2D(cout, dtype=dtype),
                    LeakyReLU(negative_slope=negative_slope),
                    MaxPool2D(kernel_size=2, stride=2),
                )
            )
        self.gap = GlobalAvgPool2D()
        self.fc = Linear(256, num_classes, bias=True)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        assert x.ndim == 4, f"Expected NCHW, got shape {x.shape}"
        assert x.shape[1] == 3, f"Expected 3 input channels, got {x.shape[1]}"
        for conv, bn, act, pool in self.blocks:
            x = conv.forward(x)
            x = bn.forward(x)
            x = act.forward(x)
            x = pool.forward(x)
        x = self.gap.forward(x)
        return self.fc.forward(x)

    def backward(self, grad_logits: cp.ndarray) -> cp.ndarray:
        assert grad_logits.ndim == 2, grad_logits.shape
        assert grad_logits.shape[1] == self.fc.out_features, (
            f"Expected (*, {self.fc.out_features}), got {grad_logits.shape}"
        )
        grad = self.fc.backward(grad_logits)
        grad = self.gap.backward(grad)
        for conv, bn, act, pool in reversed(self.blocks):
            grad = pool.backward(grad)
            grad = act.backward(grad)
            grad = bn.backward(grad)
            grad = conv.backward(grad)
        return grad

    def zero_grad(self) -> None:
        """Clear stored gradients (use before ``backward`` if grads are accumulated)."""
        for conv, bn, _, _ in self.blocks:
            conv.dW = cp.zeros_like(conv.weights)
            if conv.bias is not None:
                conv.db = cp.zeros_like(conv.bias)
            if bn.affine:
                bn.dgamma = cp.zeros_like(bn.gamma)
                bn.dbeta = cp.zeros_like(bn.beta)
        self.fc.dW = cp.zeros_like(self.fc.W)
        if self.fc.use_bias:
            self.fc.db = cp.zeros_like(self.fc.b)

    def sgd_step(self, learning_rate: float) -> None:
        """Apply one SGD update: param -= lr * grad (call after ``backward``)."""
        lr = float(learning_rate)
        for conv, bn, _, _ in self.blocks:
            assert conv.dW is not None, "Call backward() before sgd_step()"
            conv.weights -= lr * conv.dW
            if conv.bias is not None:
                assert conv.db is not None
                conv.bias -= lr * conv.db
            if bn.affine:
                bn.gamma -= lr * bn.dgamma
                bn.beta -= lr * bn.dbeta
        self.fc.W -= lr * self.fc.dW
        if self.fc.use_bias:
            self.fc.b -= lr * self.fc.db

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)
