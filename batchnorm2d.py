import cupy as cp

class BatchNorm2D:
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype=cp.float64,
    ):
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.dtype = dtype
        self.training = True

        self.running_mean = cp.zeros(self.num_features, dtype=dtype)
        self.running_var = cp.ones(self.num_features, dtype=dtype)

        if self.affine:
            self.gamma = cp.ones(self.num_features, dtype=dtype)
            self.beta = cp.zeros(self.num_features, dtype=dtype)
        else:
            self.gamma = None
            self.beta = None

        self.dgamma = None
        self.dbeta = None
        self._x = None
        self._x_hat = None
        self._std = None
        self._shape = None

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)

    def eval(self) -> None:
        self.training = False

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        assert x.ndim == 4, f"Expected (N, C, H, W), got shape {x.shape}"
        _, c, _, _ = x.shape
        assert c == self.num_features, (
            f"Input has {c} channels, expected {self.num_features}"
        )

        self._shape = x.shape
        reduce_axes = (0, 2, 3)

        if self.training:
            mean = x.mean(axis=reduce_axes)
            var = x.var(axis=reduce_axes, ddof=0)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        mean_bc = mean[None, :, None, None]
        var_bc = var[None, :, None, None]
        std = cp.sqrt(var_bc + self.eps)
        x_hat = (x - mean_bc) / std

        self._x = x
        self._x_hat = x_hat
        self._std = std

        if self.affine:
            out = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
        else:
            out = x_hat
        return out

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        assert self._x is not None, "Run forward() before backward()"
        assert grad_output.shape == self._shape

        reduce_axes = (0, 2, 3)

        if self.affine:
            gamma_bc = self.gamma[None, :, None, None]
            dx_hat = grad_output * gamma_bc
            self.dgamma = (grad_output * self._x_hat).sum(axis=reduce_axes)
            self.dbeta = grad_output.sum(axis=reduce_axes)
        else:
            dx_hat = grad_output
            self.dgamma = None
            self.dbeta = None

        x_hat = self._x_hat
        std = self._std
        mean_dx = dx_hat.mean(axis=reduce_axes, keepdims=True)
        mean_dx_xhat = (dx_hat * x_hat).mean(axis=reduce_axes, keepdims=True)
        dx = (dx_hat - mean_dx - x_hat * mean_dx_xhat) / std
        return dx

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"BatchNorm2D(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )