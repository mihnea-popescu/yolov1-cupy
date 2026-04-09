import cupy as cp

from batchnorm2d import BatchNorm2D
from conv2d import Conv2D
from global_avg_pool2d import GlobalAvgPool2D
from leaky_relu import LeakyReLU
from linear import Linear
from maxpool import MaxPool2D


class MiniDarknet:
    """
    Mini Darknet backbone + classifier head (forward only).

    Input: (N, 3, 224, 224). Output: (N, num_classes) logits.
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

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)
