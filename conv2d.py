import cupy as cp
import cupyx


def get_im2col_idx(x_shape: tuple, kernel_size: int, pad: int, stride: int):
    N, C, H, W = x_shape
    H_out = (H + 2 * pad - kernel_size) // stride + 1
    W_out = (W + 2 * pad - kernel_size) // stride + 1

    i0 = cp.repeat(cp.arange(kernel_size), kernel_size)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(H_out), W_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = cp.tile(cp.arange(kernel_size), kernel_size)
    j0 = cp.tile(j0, C)
    j1 = stride * cp.tile(cp.arange(W_out), H_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = cp.repeat(cp.arange(C), kernel_size * kernel_size).reshape(-1, 1)
    return k, i, j


def im2col(x: cp.ndarray, kernel_size: int, pad: int, stride: int) -> cp.ndarray:
    N, C, H, W = x.shape
    x_pad = cp.pad(x, ((0, 0), (0, 0), (pad, pad),
                   (pad, pad)), mode='constant')
    k, i, j = get_im2col_idx(x.shape, kernel_size, pad, stride)
    cols = x_pad[:, k, i, j]  # N, C*k*k, H_out * W_out
    cols = cols.transpose(1, 2, 0).reshape(C * kernel_size * kernel_size, -1)
    return cols


def col2im(cols: cp.ndarray, x_shape: tuple, kernel_size: int, pad: int, stride: int) -> cp.ndarray:
    N, C, H, W = x_shape

    H_padded = H + 2 * pad
    W_padded = W + 2 * pad

    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_idx(x_shape, kernel_size, pad, stride)

    # cols: (C*K*K, N*H_out*W_out)
    cols_reshaped = cols.reshape(C * kernel_size * kernel_size, -1, N)
    cols_reshaped = cols_reshaped.transpose(
        2, 0, 1)   # (N, C*K*K, H_out*W_out)

    n_idx = cp.arange(N)[:, None, None]   # (N,1,1)

    cupyx.scatter_add(
        x_padded, (n_idx, k[None, :, :], i[None, :, :], j[None, :, :]), cols_reshaped)

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:pad + H, pad:pad + W]


class Conv2D:
    """2D Convolutional layer with support for forward and backward pass

    Parameters:
    -----------
    in_channels: int
      Number of channels in the input image
    out_channels: int
      Number of channels produced by the convolution
    kernel_size: int
      Size of the convolutional kernel.
    stride: int
      Stride of the convolution. Default: 1.
    padding: int
      Padding added to all four sides of the input. Default: 0
    bias: bool
      Whether to include a bias term. Default: False
    dtype: cp.dtype
      Data type of the weights and biases. Default: cp.float64
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, dtype=cp.float64):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        self.weights = cp.zeros(
            (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
        self.bias = None
        if bias:
            self.bias = cp.zeros(out_channels, dtype=dtype)

        self.db = None
        self.dW = None

    def set_weights(self, weights: cp.ndarray):
        expected_shape = (self.out_channels, self.in_channels,
                          self.kernel_size, self.kernel_size)
        assert weights.shape == expected_shape, \
            f"Expected weights shape {expected_shape}, got {weights.shape}"
        self.weights = weights.astype(self.dtype, copy=False)

    def set_bias(self, bias: cp.ndarray):
        expected_shape = (self.out_channels,)
        assert bias.shape == expected_shape, \
            f"Expected bias shape {expected_shape}, got {bias.shape}"
        self.bias = bias.astype(self.dtype, copy=False)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Parameters
        ----------
        x : cp.ndarray of shape (N, C_in, H, W)

        Returns
        -------
        cp.ndarray of shape (N, C_out, H_out, W_out)
        """
        assert x.ndim == 4, f"Expected 4D input (N, C, H, W), got {x.ndim}D"

        N, C, H, W = x.shape
        assert C == self.in_channels, f"Input has {C} channels, expected {self.in_channels}"

        p = self.padding
        s = self.stride
        k = self.kernel_size

        assert (H + 2 * p - k) % s == 0, f"Height of input is not divisible by stride"
        assert (W + 2 * p - k) % s == 0, f"Width of input is not divisible by stride"
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        cols = im2col(x, k, pad=p, stride=s)
        weight_2d = self.weights.reshape(self.out_channels, -1)
        out_2d = weight_2d @ cols

        out = out_2d.reshape(self.out_channels, H_out,
                             W_out, N).transpose(3, 0, 1, 2)
        if self.bias is not None:
            out += self.bias[None, :, None, None]

        self.x = x
        self.cols = cols
        self.out_shape = out.shape

        return out

    def backward(self, grad_output: cp.ndarray):
        assert self.x is not None, "Run forward() before backward()"
        assert grad_output.shape == self.out_shape, \
            f"Expected grad_output shape {self.out_shape}, got {grad_output.shape}"

        N, C_out, H_out, W_out = self.out_shape

        # bias gradient
        if self.bias is not None:
            self.db = grad_output.sum(axis=(0, 2, 3))

        grad_output_2d = grad_output.transpose(1, 2, 3, 0).reshape(C_out, -1)

        # weight gradient
        dW_2d = grad_output_2d @ self.cols.T
        self.dW = dW_2d.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # input gradient
        weight_2d = self.weights.reshape(self.out_channels, -1)
        dcols = weight_2d.T @ grad_output_2d
        dx = col2im(dcols, self.x.shape, self.kernel_size,
                    self.padding, self.stride)

        return dx
