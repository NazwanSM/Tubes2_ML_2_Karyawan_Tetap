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
        self._cache_token = int(token_id)
        return self.W[int(token_id)]

    def forward_batch(self, token_ids):
        self._cache_tokens = np.asarray(token_ids, dtype=np.int32)
        return self.W[self._cache_tokens]

    def forward_sequence(self, token_ids):
        return self.W[token_ids]

    def backward(self, dout, token_ids=None):
        if token_ids is None:
            token_ids = getattr(self, '_cache_tokens',
                                getattr(self, '_cache_token', None))
        dW = np.zeros_like(self.W)
        np.add.at(dW, token_ids, dout)
        return dW


class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.W_x = np.asarray(kernel,           dtype=np.float32)
        self.W_h = np.asarray(recurrent_kernel, dtype=np.float32)
        self.b = np.asarray(bias,             dtype=np.float32)
        self._cache = {}

    @property
    def units(self):
        return self.W_h.shape[0]

    def forward(self, x, h):
        h_new = np.tanh(x @ self.W_x + h @ self.W_h + self.b)
        self._cache = {'x': x, 'h': h, 'h_new': h_new}
        return h_new

    def backward(self, dh_new):
        x, h, h_new = self._cache['x'], self._cache['h'], self._cache['h_new']

        # tanh backward
        dz = dh_new * (1.0 - h_new ** 2)

        dx = dz @ self.W_x.T
        dh_prev = dz @ self.W_h.T

        single = x.ndim == 1
        if single:
            dW_x = np.outer(x, dz)
            dW_h = np.outer(h, dz)
            db = dz
        else:
            dW_x = x.T @ dz
            dW_h = h.T @ dz
            db = dz.sum(axis=0)

        return dx, dh_prev, dW_x, dW_h, db

    def zero_state(self, batch_size=1):
        return np.zeros((batch_size, self.units), dtype=np.float32)


class LSTMCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.W_x = np.asarray(kernel,           dtype=np.float32)
        self.W_h = np.asarray(recurrent_kernel, dtype=np.float32)
        b = np.asarray(bias, dtype=np.float32)
        self.b = b[0] + b[1] if b.ndim == 2 else b
        self._cache = {}

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

        self._cache = {
            'x': x, 'h': h, 'c': c,
            'i': i, 'f': f, 'g': g, 'o': o,
            'c_new': c_new, 'h_new': h_new,
        }
        return h_new, c_new

    def backward(self, dh_new, dc_new):
        x, h, c_prev = self._cache['x'], self._cache['h'], self._cache['c']
        i, f, g, o = self._cache['i'], self._cache['f'], self._cache['g'], self._cache['o']
        c_new = self._cache['c_new']

        tanh_c_new = np.tanh(c_new)

        # h_new = o * tanh(c_new)
        do = dh_new * tanh_c_new
        dc_from_h = dh_new * o * (1.0 - tanh_c_new ** 2)
        dc = dc_new + dc_from_h        

        # c_new = f * c_prev + i * g
        di      = dc * g
        df      = dc * c_prev
        dg      = dc * i
        dc_prev = dc * f

        # Gate activation backward
        di_raw = di * i * (1.0 - i)    
        df_raw = df * f * (1.0 - f)    
        dg_raw = dg * (1.0 - g ** 2)   
        do_raw = do * o * (1.0 - o)    

        dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=-1)

        dx = dgates @ self.W_x.T
        dh_prev = dgates @ self.W_h.T

        single = x.ndim == 1
        if single:
            dW_x = np.outer(x, dgates)
            dW_h = np.outer(h, dgates)
            db = dgates
        else:
            dW_x = x.T @ dgates
            dW_h = h.T @ dgates
            db = dgates.sum(axis=0)

        return dx, dh_prev, dc_prev, dW_x, dW_h, db

    def zero_state(self, batch_size=1):
        z = np.zeros((batch_size, self.units), dtype=np.float32)
        return z.copy(), z.copy()


class DenseLayer:
    def __init__(self, kernel, bias, activation=None):
        self.W = np.asarray(kernel, dtype=np.float32)
        self.b = np.asarray(bias,   dtype=np.float32)
        self.activation = activation   
        self._cache = {}

    def forward(self, x):
        self._cache['x'] = x
        z = x @ self.W + self.b
        self._cache['z'] = z
        if self.activation == "softmax":
            z2 = z - z.max(axis=-1, keepdims=True)
            e  = np.exp(z2)
            out = e / e.sum(axis=-1, keepdims=True)
        elif self.activation == "relu":
            out = np.maximum(0, z)
        else:
            out = z
        self._cache['out'] = out
        return out

    def backward(self, dout):
        x   = self._cache['x']
        z   = self._cache['z']
        out = self._cache['out']

        if self.activation == "softmax":
            s = out
            dz = s * (dout - (dout * s).sum(axis=-1, keepdims=True))
        elif self.activation == "relu":
            dz = dout * (z > 0)
        else:
            dz = dout

        single = x.ndim == 1
        if single:
            x  = x[np.newaxis]
            dz = dz[np.newaxis]

        dx = dz @ self.W.T
        dW = x.T @ dz
        db = dz.sum(axis=0)

        return (dx[0] if single else dx), dW, db
