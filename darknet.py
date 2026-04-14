from pathlib import Path

import cupy as cp
import numpy as np

from conv2d import Conv2D
from batchnorm2d import BatchNorm2D
from global_avg_pool2d import GlobalAvgPool2D
from leaky_relu import LeakyReLU
from linear import Linear
from maxpool import MaxPool2D


class Darknet:
    """
    Darknet + classficiation head

    Input:  (N, 3, 224, 224)
    Output: (N, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dtype=cp.float32,
        negative_slope: float = 0.1,
    ):
        self.dtype = dtype
        self.num_classes = num_classes

        self.layers = [
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
        
        self.gap = GlobalAvgPool2D()
        self.fc = Linear(1024, num_classes, bias=True)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        assert x.ndim == 4, f"Expected NCHW, got shape {x.shape}"
        assert x.shape[1] == 3, f"Expected 3 input channels, got {x.shape[1]}"

        for layer in self.layers:
            x = layer.forward(x)

        x = self.gap.forward(x)
        return self.fc.forward(x)

    def backward(self, grad_logits: cp.ndarray) -> cp.ndarray:
        assert grad_logits.ndim == 2, f"Expected 2D logits grad, got {grad_logits.shape}"
        assert grad_logits.shape[1] == self.fc.out_features, (
            f"Expected (*, {self.fc.out_features}), got {grad_logits.shape}"
        )

        grad = self.fc.backward(grad_logits)
        grad = self.gap.backward(grad)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.dW = cp.zeros_like(layer.weights)
                if layer.bias is not None:
                    layer.db = cp.zeros_like(layer.bias)
            
            elif isinstance(layer, BatchNorm2D):
                if layer.affine:
                    layer.dgamma = cp.zeros_like(layer.gamma)
                    layer.dbeta = cp.zeros_like(layer.beta)
            
            elif isinstance(layer, LeakyReLU):
                continue
            
            elif isinstance(layer, MaxPool2D):
                continue
                
            else:
                raise NotImplementedError("Unsupported layer")

        self.fc.dW = cp.zeros_like(self.fc.W)
        if self.fc.use_bias:
            self.fc.db = cp.zeros_like(self.fc.b)

    def sgd_step(self, learning_rate: float):
        lr = float(learning_rate)

        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.weights -= lr * layer.dW
                if layer.bias is not None:
                    layer.bias -= lr * layer.db

            elif isinstance(layer, BatchNorm2D):
                if layer.affine:
                    layer.gamma -= lr * layer.dgamma
                    layer.beta -= lr * layer.dbeta
                    
            elif isinstance(layer, LeakyReLU):
                continue
                
            elif isinstance(layer, MaxPool2D):
                continue
            
            else:
                raise NotImplementedError("Unsupported layer")

        self.fc.W -= lr * self.fc.dW
        if self.fc.use_bias:
            self.fc.b -= lr * self.fc.db

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "num_classes": np.array([self.fc.out_features], dtype=np.int64),
        }

        conv_idx = 0
        bn_idx = 0

        for layer in self.layers:
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

        payload["fc_W"] = cp.asnumpy(self.fc.W)
        if self.fc.use_bias:
            payload["fc_b"] = cp.asnumpy(self.fc.b)

        np.savez_compressed(path, **payload)

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        nc = int(data["num_classes"][0])
        assert nc == self.fc.out_features, (
            f"Checkpoint has num_classes={nc}, model has {self.fc.out_features}"
        )

        conv_idx = 0
        bn_idx = 0

        for layer in self.layers:
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

        self.fc.W = cp.asarray(data["fc_W"], dtype=self.dtype)
        if self.fc.use_bias:
            self.fc.b = cp.asarray(data["fc_b"], dtype=self.dtype)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)