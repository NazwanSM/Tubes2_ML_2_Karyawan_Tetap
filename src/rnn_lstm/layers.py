import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class EmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.W = np.asarray(embedding_matrix, dtype=np.float32)

    @property
    def embed_dim(self):
        return self.W.shape[1]

    def forward(self, token_id):
        return self.W[int(token_id)]

    def forward_batch(self, token_ids):
        return self.W[np.asarray(token_ids, dtype=np.int32)]

    def forward_sequence(self, token_ids):
        return self.W[token_ids]

class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.W_x = np.asarray(kernel,           dtype=np.float32)
        self.W_h = np.asarray(recurrent_kernel, dtype=np.float32)
        self.b = np.asarray(bias,             dtype=np.float32)

    @property
    def units(self):
        return self.W_h.shape[0]

    def forward(self, x, h):
        return np.tanh(x @ self.W_x + h @ self.W_h + self.b)

    def zero_state(self, batch_size=1):
        if batch_size == 1:
            return np.zeros(self.units, dtype=np.float32)
        return np.zeros((batch_size, self.units), dtype=np.float32)

class LSTMCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.W_x = np.asarray(kernel,           dtype=np.float32)
        self.W_h = np.asarray(recurrent_kernel, dtype=np.float32)
        b = np.asarray(bias, dtype=np.float32)
        self.b = b[0] + b[1] if b.ndim == 2 else b

    @property
    def units(self):
        return self.W_h.shape[0]

    def forward(self, x, h, c):
        gates = x @ self.W_x + h @ self.W_h + self.b
        i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=-1)

        i = _sigmoid(i_gate)
        f = _sigmoid(f_gate)
        g = np.tanh(g_gate)
        o = _sigmoid(o_gate)

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    def zero_state(self, batch_size=1):
        if batch_size == 1:
            z = np.zeros(self.units, dtype=np.float32)
            return z.copy(), z.copy()
        z = np.zeros((batch_size, self.units), dtype=np.float32)
        return z.copy(), z.copy()


class DenseLayer:
    def __init__(self, kernel, bias, activation=None):
        self.W = np.asarray(kernel, dtype=np.float32)
        self.b = np.asarray(bias,   dtype=np.float32)
        self.activation = activation

    def forward(self, x):
        out = x @ self.W + self.b
        if self.activation == "softmax":
            out = out - out.max(axis=-1, keepdims=True)
            e = np.exp(out)
            out = e / e.sum(axis=-1, keepdims=True)
        elif self.activation == "relu":
            out = np.maximum(0, out)
        return out
