import cupy as cp

class AvgPool2d:
    # 2D Average Pooling layer.
    # Equivalent to torch.nn.AvgPool2d.

    # Slides a window over each feature map and computes the average
    # of all values inside that window.

    # Parameters:
    # kernel_size  : int or tuple  - size of the pooling window
    # stride       : int or tuple  - step size between windows (default: same as kernel_size)
    # padding      : int or tuple  - zero-padding added to both sides of input (default: 0)


    def __init__(self, kernel_size, stride=None, padding=0):
        # Allow either an int or a (height, width) tuple for each parameter
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = self.kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self._input_cache = None  # saved for backward pass

    # Forward pass
    # input shape:  (batch, channels, H, W)
    # output shape: (batch, channels, H_out, W_out)
    # H_out = floor((H + 2*padding - kernel_size) / stride) + 1
    # W_out = floor((W + 2*padding - kernel_size) / stride) + 1
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        batch, channels, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # Pad input with zeros if needed
        if pH > 0 or pW > 0:
            x_padded = cp.pad(x, ((0,0), (0,0), (pH,pH), (pW,pW)), mode='constant')
        else:
            x_padded = x

        self._input_cache = x_padded

        # Compute output dimensions
        H_out = (H + 2*pH - kH) // sH + 1
        W_out = (W + 2*pW - kW) // sW + 1

        out = cp.zeros((batch, channels, H_out, W_out), dtype=x.dtype)

        # Slide the window across height and width
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                w_start = j * sW
                window = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                out[:, :, i, j] = cp.mean(window, axis=(2, 3))

        return out

    # Backward pass
    # d_out shape: (batch, channels, H_out, W_out) — gradient from next layer
    # returns: (batch, channels, H, W) — gradient to pass backward
    #
    # Each input element contributed equally to the window average,
    # so the gradient is distributed evenly: d_input = d_out / (kH * kW)
    def backward(self, d_out: cp.ndarray) -> cp.ndarray:
        x_padded = self._input_cache
        batch, channels, H_pad, W_pad = x_padded.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        _, _, H_out, W_out = d_out.shape

        d_input_padded = cp.zeros_like(x_padded)
        pool_area = kH * kW

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                w_start = j * sW
                # Distribute gradient equally across every element in the window
                d_input_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += (d_out[:, :, i, j][:, :, cp.newaxis, cp.newaxis] / pool_area)

        # Strip padding to return gradient in original input shape
        if pH > 0 or pW > 0:
            d_input = d_input_padded[:, :, pH:-pH, pW:-pW]
        else:
            d_input = d_input_padded

        return d_input

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def __repr__(self):
        return (f"AvgPool2d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")