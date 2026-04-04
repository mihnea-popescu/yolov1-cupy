import numpy as np

class Linear:
    """
    Fully connected (dense) layer.
    Equivalent to torch.nn.Linear(in_features, out_features).

    Parameters
    ----------
    in_features  : int   - size of each input sample
    out_features : int   - size of each output sample
    bias         : bool  - if True, adds a learnable bias (default: True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weight initialization (Kaiming / He uniform, same as PyTorch default)
        # Keeps variance stable across layers, especially with ReLU/LeakyReLU
        limit = np.sqrt(1.0 / in_features)
        self.W = np.random.uniform(-limit, limit, (out_features, in_features))  # shape: (out, in)
        self.b = np.zeros(out_features) if bias else None  # shape: (out,)

        # Gradients (populated during backward pass)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._input_cache = None  # saved for backprop

    
    # Forward pass:  output = input @ W.T + b
    # input shape:  (batch_size, in_features)
    # output shape: (batch_size, out_features)
    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == self.in_features, (f"Expected input with {self.in_features} features, got {x.shape[-1]}")
        self._input_cache = x
        out = x @ self.W.T          # (batch, out_features)
        if self.use_bias:
            out += self.b           # broadcast over batch dimension
        return out

    
    # Backward pass: compute gradients w.r.t. W, b, and input
    # d_out shape: (batch_size, out_features)  — gradient from next layer
    # returns:     (batch_size, in_features)   — gradient to pass backward
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x = self._input_cache
        self.dW = d_out.T @ x           # (out_features, in_features)
        if self.use_bias:
            self.db = d_out.sum(axis=0) # sum over batch
        d_input = d_out @ self.W        # (batch_size, in_features)
        return d_input

    
    # Convenience: lets you call the layer like a function, e.g. layer(x)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, " f"out_features={self.out_features}, bias={self.use_bias})")