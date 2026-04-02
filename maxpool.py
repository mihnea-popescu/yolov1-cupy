import cupy as cp
import cupyx
from typing import Any

class MaxPool2D:
    """2D Max Pooling layer with support for forward and backward passes.

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the pooling window.
    stride : int or tuple[int, int] or None
        Stride of the pooling window. Defaults to kernel_size.
    padding : int or tuple[int, int]
        Zero-padding added to both sides of the input.
    """

    def __init__(self, kernel_size=2, stride=None, padding=0):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple[Any, ...](kernel_size)
        self.stride = self.kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple[Any, ...](stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple[Any, ...](padding)

        self._input_shape = None
        self._max_indices = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Parameters
        ----------
        x : cp.ndarray of shape (N, C, H, W)

        Returns
        -------
        cp.ndarray of shape (N, C, H_out, W_out)
        """
        assert x.ndim == 4, f"Expected 4D input (N, C, H, W), got {x.ndim}D"
        self._input_shape = x.shape
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x = cp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                        mode="constant", constant_values=-cp.inf)

        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        # Build strided view of all pooling windows: (N, C, H_out, W_out, kh, kw)
        strides = x.strides
        windows = cp.lib.stride_tricks.as_strided(
            x,
            shape=(n, c, h_out, w_out, kh, kw),
            strides=(strides[0], strides[1],
                     strides[2] * sh, strides[3] * sw,
                     strides[2], strides[3]),
        )

        out = windows.reshape(n, c, h_out, w_out, -1).max(axis=-1)

        flat_idx = windows.reshape(n, c, h_out, w_out, -1).argmax(axis=-1)
        self._max_indices = flat_idx

        return out

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        Parameters
        ----------
        grad_output : cp.ndarray of shape (N, C, H_out, W_out)

        Returns
        -------
        grad_input : cp.ndarray of shape (N, C, H, W) — same shape as the forward input.
        """
        assert self._input_shape is not None, "Call forward() before backward()"
        n, c, h, w = self._input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        h_padded = h + 2 * ph
        w_padded = w + 2 * pw
        grad_padded = cp.zeros((n, c, h_padded, w_padded), dtype=grad_output.dtype)

        h_out, w_out = grad_output.shape[2], grad_output.shape[3]

        row_offsets = self._max_indices // kw
        col_offsets = self._max_indices % kw

        nn = cp.arange(n)[:, None, None, None]
        cc = cp.arange(c)[None, :, None, None]
        hh = cp.arange(h_out)[None, None, :, None]
        ww = cp.arange(w_out)[None, None, None, :]

        h_idx = hh * sh + row_offsets
        w_idx = ww * sw + col_offsets

        cupyx.scatter_add(grad_padded, (nn, cc, h_idx, w_idx), grad_output)

        if ph > 0 or pw > 0:
            return grad_padded[:, :, ph:ph + h, pw:pw + w]
        return grad_padded