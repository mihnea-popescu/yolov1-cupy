import cupy as cp


class Dropout:
    """
    Inverted dropout with train/eval modes.

    Training:
      mask  = Bernoulli(1 - p) / (1 - p)
      out   = x * mask
      dx    = d_out * mask

    Evaluation:
      out   = x
      dx    = d_out
    """

    def __init__(self, p: float = 0.5, dtype=cp.float32):
        assert 0.0 <= p < 1.0, f"Dropout probability must be in [0, 1), got {p}"
        self.p = float(p)
        self.dtype = dtype
        self.training = True
        self._mask = None

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)

    def eval(self) -> None:
        self.training = False

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.p
        mask = (cp.random.random(x.shape) < keep_prob).astype(self.dtype) / keep_prob
        self._mask = mask
        return x * mask

    def backward(self, d_out: cp.ndarray) -> cp.ndarray:
        if self._mask is None:
            return d_out
        return d_out * self._mask

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
