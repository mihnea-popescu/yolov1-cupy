from pathlib import Path

import cupy as cp
import numpy as np

from conv2d import Conv2D
from batchnorm2d import BatchNorm2D
from global_avg_pool2d import GlobalAvgPool2D
from leaky_relu import LeakyReLU
from linear import Linear
from maxpool import MaxPool2D
from flatten import Flatten
from dropout import Dropout


class YOLO:
    """
    YOLOv1

    Input:  (N, 3, 448, 448)
    Output: (N, 7, 7, 30) logits
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dtype=cp.float32,
        negative_slope: float = 0.1,
    ):
        self.dtype = dtype
        self.num_classes = num_classes

        self.backbone = [
          Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias=False, dtype=dtype),
          BatchNorm2D(64, dtype=dtype),
          LeakyReLU(negative_slope),
          MaxPool2D(2, 2),

          Conv2D(64, 192, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(192, dtype=dtype),
          LeakyReLU(negative_slope),
          MaxPool2D(2, 2),

          Conv2D(192, 128, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(128, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(128, 256, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 256, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),
          MaxPool2D(2, 2),

          Conv2D(512, 256, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 256, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 256, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 256, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(256, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 512, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(1024, dtype=dtype),
          LeakyReLU(negative_slope),
          MaxPool2D(2, 2),

          Conv2D(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(1024, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype),
          BatchNorm2D(512, dtype=dtype),
          LeakyReLU(negative_slope),

          Conv2D(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
          BatchNorm2D(1024, dtype=dtype),
          LeakyReLU(negative_slope),
        ]
        
        self.head = [
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
            BatchNorm2D(1024, dtype=dtype),
            LeakyReLU(negative_slope),
            
            Conv2D(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False, dtype=dtype),
            BatchNorm2D(1024, dtype=dtype),
            LeakyReLU(negative_slope),
            
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
            BatchNorm2D(1024, dtype=dtype),
            LeakyReLU(negative_slope),
            
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
            BatchNorm2D(1024, dtype=dtype),
            LeakyReLU(negative_slope),
            
            Flatten(),

            Linear(50176, 4096, bias=False),
            Dropout(p=0.5),
            LeakyReLU(negative_slope),

            Linear(4096, 7 * 7 * 30, bias=False),
        ]

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        assert x.ndim == 4, f"Expected NCHW, got shape {x.shape}"
        assert x.shape[1] == 3, f"Expected 3 input channels, got {x.shape[1]}"
        for layer in self.backbone:
            x = layer.forward(x)
        for layer in self.head:
            print(type(layer), x.shape)
            x = layer.forward(x)
        return x

    def backward(self, grad_logits: cp.ndarray) -> cp.ndarray:
        assert grad_logits.ndim == 2, f"Expected 2D logits grad, got {grad_logits.shape}"
        final_out_features = self.head[-1].out_features
        assert grad_logits.shape[1] == final_out_features, (
            f"Expected (*, {final_out_features}), got {grad_logits.shape}"
        )

        grad = grad_logits
        for layer in reversed(self.head):
            grad = layer.backward(grad)
        for layer in reversed(self.backbone):
            grad = layer.backward(grad)

        return grad

    def _zero_grad_helper(self, layer):
        if isinstance(layer, Conv2D):
            layer.dW = cp.zeros_like(layer.weights)
            if layer.bias is not None:
                layer.db = cp.zeros_like(layer.bias)
        
        elif isinstance(layer, BatchNorm2D):
            if layer.affine:
                layer.dgamma = cp.zeros_like(layer.gamma)
                layer.dbeta = cp.zeros_like(layer.beta)
        
        elif isinstance(layer, Linear):
            layer.dW = cp.zeros_like(layer.W)
            if layer.use_bias:
                layer.db = cp.zeros_like(layer.b)
                
        elif isinstance(layer, Flatten):
            pass

        elif isinstance(layer, Dropout):
            pass

        elif isinstance(layer, LeakyReLU):
            pass

        elif isinstance(layer, MaxPool2D):
            pass

        else:
            raise NotImplementedError("Unsupported layer")

    def zero_grad(self):
        for layer in self.backbone:
            self._zero_grad_helper(layer)
        for layer in self.head:
            self._zero_grad_helper(layer)

    def _sgd_step_helper(self, layer, lr):
        if isinstance(layer, Conv2D):
            layer.weights -= lr * layer.dW
            if layer.bias is not None:
                layer.bias -= lr * layer.db

        elif isinstance(layer, BatchNorm2D):
            if layer.affine:
                layer.gamma -= lr * layer.dgamma
                layer.beta -= lr * layer.dbeta
                
        elif isinstance(layer, Linear):
            layer.W -= lr * layer.dW
            if layer.use_bias:
                layer.b -= lr * layer.db

        elif isinstance(layer, Flatten):
            pass

        elif isinstance(layer, Dropout):
            pass

        elif isinstance(layer, LeakyReLU):
            pass

        elif isinstance(layer, MaxPool2D):
            pass

        else:
            raise NotImplementedError("Unsupported layer")

    def sgd_step(self, learning_rate: float):
        lr = float(learning_rate)
        for layer in self.backbone:
            self._sgd_step_helper(layer, lr)
        for layer in self.head:
            self._sgd_step_helper(layer, lr)

    def init_optimizer(self) -> None:
        """Allocate zero-initialized momentum (velocity) buffers for all trainable params.

        Must be called once before the first `sgd_momentum_step`.
        """
        for layer in list(self.backbone) + list(self.head):
            if isinstance(layer, Conv2D):
                layer.vW = cp.zeros_like(layer.weights)
                if layer.bias is not None:
                    layer.vb = cp.zeros_like(layer.bias)
            elif isinstance(layer, BatchNorm2D):
                if layer.affine:
                    layer.vgamma = cp.zeros_like(layer.gamma)
                    layer.vbeta = cp.zeros_like(layer.beta)
            elif isinstance(layer, Linear):
                layer.vW = cp.zeros_like(layer.W)
                if layer.use_bias:
                    layer.vb = cp.zeros_like(layer.b)

    def _sgd_momentum_step_helper(
        self,
        layer,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        if isinstance(layer, Conv2D):
            layer.vW = momentum * layer.vW + layer.dW + weight_decay * layer.weights
            layer.weights -= lr * layer.vW
            if layer.bias is not None:
                layer.vb = momentum * layer.vb + layer.db
                layer.bias -= lr * layer.vb

        elif isinstance(layer, BatchNorm2D):
            if layer.affine:
                layer.vgamma = momentum * layer.vgamma + layer.dgamma + weight_decay * layer.gamma
                layer.gamma -= lr * layer.vgamma
                layer.vbeta = momentum * layer.vbeta + layer.dbeta + weight_decay * layer.beta
                layer.beta -= lr * layer.vbeta

        elif isinstance(layer, Linear):
            layer.vW = momentum * layer.vW + layer.dW + weight_decay * layer.W
            layer.W -= lr * layer.vW
            if layer.use_bias:
                layer.vb = momentum * layer.vb + layer.db
                layer.b -= lr * layer.vb

        elif isinstance(layer, (Flatten, Dropout, LeakyReLU, MaxPool2D)):
            pass

        else:
            raise NotImplementedError("Unsupported layer")

    def sgd_momentum_step(
        self,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ) -> None:
        """SGD with Polyak momentum and L2 weight decay (paper recipe).

        v = momentum * v + dW + weight_decay * W
        W -= lr * v

        Weight decay applies to conv weights, BN gamma/beta, and linear W.
        It is not applied to biases (convention).
        """
        lr = float(learning_rate)
        m = float(momentum)
        wd = float(weight_decay)
        for layer in self.backbone:
            self._sgd_momentum_step_helper(layer, lr, m, wd)
        for layer in self.head:
            self._sgd_momentum_step_helper(layer, lr, m, wd)

    def train(self, mode: bool = True) -> None:
        """Put BN/Dropout into training mode."""
        for layer in list(self.backbone) + list(self.head):
            if isinstance(layer, (BatchNorm2D, Dropout)):
                layer.train(mode)

    def eval(self) -> None:
        """Put BN/Dropout into evaluation mode."""
        for layer in list(self.backbone) + list(self.head):
            if isinstance(layer, (BatchNorm2D, Dropout)):
                layer.eval()

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        payload = {}
        conv_idx = 0
        bn_idx = 0
        fc_idx = 0
        def save_layer(layer):
            nonlocal conv_idx, bn_idx, fc_idx

            if isinstance(layer, Conv2D):
                payload[f"conv{conv_idx}_weights"] = cp.asnumpy(layer.weights)
                if layer.bias is not None:
                    payload[f"conv{conv_idx}_bias"] = cp.asnumpy(layer.bias)
                conv_idx += 1
            elif isinstance(layer, BatchNorm2D):
                if layer.affine:
                    payload[f"bn{bn_idx}_gamma"] = cp.asnumpy(layer.gamma)
                    payload[f"bn{bn_idx}_beta"] = cp.asnumpy(layer.beta)
                payload[f"bn{bn_idx}_running_mean"] = cp.asnumpy(layer.running_mean)
                payload[f"bn{bn_idx}_running_var"] = cp.asnumpy(layer.running_var)
                bn_idx += 1
            elif isinstance(layer, Linear):
                payload[f"fc{fc_idx}_W"] = cp.asnumpy(layer.W)
                if layer.use_bias:
                    payload[f"fc{fc_idx}_b"] = cp.asnumpy(layer.b)
                fc_idx += 1

        for layer in self.backbone:
            save_layer(layer)
        for layer in self.head:
            save_layer(layer)

        np.savez_compressed(path, **payload)

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        conv_idx = 0
        bn_idx = 0
        fc_idx = 0
        def load_layer(layer):
            nonlocal conv_idx, bn_idx, fc_idx
            if isinstance(layer, Conv2D):
                layer.weights = cp.asarray(data[f"conv{conv_idx}_weights"], dtype=self.dtype)

                key_b = f"conv{conv_idx}_bias"
                if layer.bias is not None:
                    if key_b not in data.files:
                        raise KeyError(f"Missing {key_b} in checkpoint")
                    layer.bias = cp.asarray(data[key_b], dtype=self.dtype)

                conv_idx += 1

            elif isinstance(layer, BatchNorm2D):
                if layer.affine:
                    layer.gamma = cp.asarray(data[f"bn{bn_idx}_gamma"], dtype=self.dtype)
                    layer.beta = cp.asarray(data[f"bn{bn_idx}_beta"], dtype=self.dtype)

                layer.running_mean = cp.asarray(data[f"bn{bn_idx}_running_mean"], dtype=self.dtype)
                layer.running_var = cp.asarray(data[f"bn{bn_idx}_running_var"], dtype=self.dtype)

                bn_idx += 1
            elif isinstance(layer, Linear):
                layer.W = cp.asarray(data[f"fc{fc_idx}_W"], dtype=self.dtype)
                if layer.use_bias:
                    layer.b = cp.asarray(data[f"fc{fc_idx}_b"], dtype=self.dtype)
                fc_idx += 1

        for layer in self.backbone:
            load_layer(layer)
        for layer in self.head:
            load_layer(layer)


    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)