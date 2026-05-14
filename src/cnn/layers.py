import math
import numpy as np

# Fungsi Aktivasi
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

_ACTIVATIONS = {
    "relu": relu,
    "softmax": softmax,
    "sigmoid": sigmoid,
    "tanh": np.tanh,
    "linear": lambda x: x,
}

def get_activation(name):
    if name is None or name == "linear":
        return None
    fn = _ACTIVATIONS.get(name)
    if fn is None:
        raise ValueError(f"Aktivasi '{name}' belum di-support. Pilihan: {list(_ACTIVATIONS)}")
    return fn


# Fungsi Helper
def _same_pad(H, W, kH, kW, stride_h, stride_w):
    out_h = math.ceil(H / stride_h)
    out_w = math.ceil(W / stride_w)
    pad_h = max((out_h - 1) * stride_h + kH - H, 0)
    pad_w = max((out_w - 1) * stride_w + kW - W, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2

def _col2im(dcols, N, H_pad, W_pad, C, kH, kW, stride_h, stride_w, out_h, out_w):
    dx_pad = np.zeros((N, H_pad, W_pad, C), dtype=np.float32)
    dcols_5d = dcols.reshape(N, out_h, out_w, kH, kW, C)
    for i in range(out_h):
        for j in range(out_w):
            h0, w0 = i * stride_h, j * stride_w
            dx_pad[:, h0:h0+kH, w0:w0+kW, :] += dcols_5d[:, i, j, :, :, :]
    return dx_pad

def _act_backward(dout, act_name, cache_z, cache_out):
    if act_name is None or act_name == 'linear':
        return dout
    if act_name == 'relu':
        return dout * (cache_z > 0)
    if act_name == 'softmax':
        s = cache_out
        return s * (dout - (dout * s).sum(axis=-1, keepdims=True))
    if act_name == 'sigmoid':
        s = cache_out
        return dout * s * (1 - s)
    if act_name == 'tanh':
        return dout * (1 - cache_out ** 2)
    return dout


# Layer Implementations
class Conv2DLayer:
    def __init__(self, kernel, bias, strides=(1, 1), padding="valid", activation=None):
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias,   dtype=np.float32)
        self.stride_h, self.stride_w = strides
        self.padding = padding.lower()
        self._act_name = activation
        self.activation = get_activation(activation)
        self.kH, self.kW = self.kernel.shape[:2]
        self._cache = {}

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        pt = pb = pl = pr = 0

        if self.padding == "same":
            pt, pb, pl, pr = _same_pad(H, W, self.kH, self.kW, self.stride_h, self.stride_w)
            x = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
            N, H, W = x.shape[:3]

        out_h = (H - self.kH) // self.stride_h + 1
        out_w = (W - self.kW) // self.stride_w + 1
        C_out = self.kernel.shape[-1]

        patch_shape = (N, out_h, out_w, self.kH, self.kW, C)
        patch_strides = (
            x.strides[0],
            x.strides[1] * self.stride_h,
            x.strides[2] * self.stride_w,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=patch_shape, strides=patch_strides)
        cols = patches.reshape(N, out_h * out_w, -1)
        W_flat = self.kernel.reshape(-1, C_out)
        z = (cols @ W_flat + self.bias).reshape(N, out_h, out_w, C_out)
        out = self.activation(z) if self.activation else z

        self._cache = {
            'x_pad': x, 'cols': cols, 'z': z, 'out': out,
            'N': N, 'H': H, 'W': W, 'C': C,
            'out_h': out_h, 'out_w': out_w,
            'pt': pt, 'pb': pb, 'pl': pl, 'pr': pr, 'single': single,
        }
        return out[0] if single else out

    def backward(self, dout):
        c = self._cache
        single = c['single']
        if single:
            dout = dout[np.newaxis]

        N, out_h, out_w = c['N'], c['out_h'], c['out_w']
        C_out = self.kernel.shape[-1]
        C_in = c['C']

        dz = _act_backward(dout, self._act_name, c['z'], c['out'])
        dz_2d = dz.reshape(N * out_h * out_w, C_out)
        cols_2d = c['cols'].reshape(N * out_h * out_w, -1)

        db = dz.sum(axis=(0, 1, 2))
        dW = (cols_2d.T @ dz_2d).reshape(self.kernel.shape)
        dcols = (dz_2d @ self.kernel.reshape(-1, C_out).T).reshape(N, out_h * out_w, -1)

        dx_pad = _col2im(dcols, N, c['H'], c['W'], C_in,
                         self.kH, self.kW, self.stride_h, self.stride_w,
                         out_h, out_w)

        pt, pb, pl, pr = c['pt'], c['pb'], c['pl'], c['pr']
        if pt or pb or pl or pr:
            H_orig = c['H'] - pt - pb
            W_orig = c['W'] - pl - pr
            dx = dx_pad[:, pt:pt+H_orig, pl:pl+W_orig, :]
        else:
            dx = dx_pad

        return (dx[0] if single else dx), dW, db


class LocallyConnected2DLayer:
    def __init__(self, kernel, bias, kernel_size, strides=(1, 1), padding="valid", activation=None):
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias,   dtype=np.float32)
        ks = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.kH, self.kW = ks
        self.stride_h, self.stride_w = strides
        self.padding = padding.lower()
        self._act_name = activation
        self.activation = get_activation(activation)
        self._cache = {}

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        sh, sw = self.stride_h, self.stride_w

        if self.padding == "same":
            pt, pb, pl, pr = _same_pad(H, W, self.kH, self.kW, sh, sw)
            x = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
            N, H, W = x.shape[:3]

        out_h = (H - self.kH) // sh + 1
        out_w = (W - self.kW) // sw + 1
        C_out = self.kernel.shape[-1]
        z = np.zeros((N, out_h, out_w, C_out), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i*sh:i*sh+self.kH, j*sw:j*sw+self.kW, :]
                z[:, i, j] = patch.reshape(N, -1) @ self.kernel[i, j] + self.bias[i, j]

        out = self.activation(z) if self.activation else z
        self._cache = {'x_pad': x, 'z': z, 'out': out, 'N': N, 'H': H, 'W': W, 'C': C,
                       'out_h': out_h, 'out_w': out_w, 'single': single}
        return out[0] if single else out

    def backward(self, dout):
        c = self._cache
        single = c['single']
        if single:
            dout = dout[np.newaxis]

        N, out_h, out_w = c['N'], c['out_h'], c['out_w']
        sh, sw = self.stride_h, self.stride_w
        x_pad = c['x_pad']
        C_in = c['C']

        dz = _act_backward(dout, self._act_name, c['z'], c['out'])
        dW = np.zeros_like(self.kernel)
        db = np.zeros_like(self.bias)
        dx_pad = np.zeros_like(x_pad)

        for i in range(out_h):
            for j in range(out_w):
                patch = x_pad[:, i*sh:i*sh+self.kH, j*sw:j*sw+self.kW, :]
                patch_flat = patch.reshape(N, -1) # (N, kH*kW*C_in)
                dz_ij = dz[:, i, j, :] # (N, C_out)

                dW[i, j] = patch_flat.T @ dz_ij # (kH*kW*C_in, C_out)
                db[i, j] = dz_ij.sum(axis=0)

                dpatch = (dz_ij @ self.kernel[i, j].T).reshape(N, self.kH, self.kW, C_in)
                dx_pad[:, i*sh:i*sh+self.kH, j*sw:j*sw+self.kW, :] += dpatch

        return (dx_pad[0] if single else dx_pad), dW, db


class MaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pH, self.pW = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size
        self._cache = {}

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        out_h = (H - self.pH) // self.stride_h + 1
        out_w = (W - self.pW) // self.stride_w + 1

        win_shape = (N, out_h, out_w, self.pH, self.pW, C)
        win_strides = (
            x.strides[0],
            x.strides[1] * self.stride_h,
            x.strides[2] * self.stride_w,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)
        out = windows.max(axis=(3, 4))

        self._cache = {'x': x, 'out_h': out_h, 'out_w': out_w, 'single': single}
        return out[0] if single else out

    def backward(self, dout):
        c = self._cache
        single = c['single']
        if single:
            dout = dout[np.newaxis]

        x = c['x']
        out_h = c['out_h']
        out_w = c['out_w']
        dx = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                h0, w0 = i * self.stride_h, j * self.stride_w
                window = x[:, h0:h0+self.pH, w0:w0+self.pW, :]
                max_vals = window.max(axis=(1, 2), keepdims=True)
                mask = (window == max_vals).astype(np.float32)
                mask /= mask.sum(axis=(1, 2), keepdims=True)
                dx[:, h0:h0+self.pH, w0:w0+self.pW, :] += mask * dout[:, i:i+1, j:j+1, :]

        return dx[0] if single else dx


class AvgPooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pH, self.pW = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size
        self._cache = {}

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        out_h = (H - self.pH) // self.stride_h + 1
        out_w = (W - self.pW) // self.stride_w + 1

        win_shape = (N, out_h, out_w, self.pH, self.pW, C)
        win_strides = (
            x.strides[0],
            x.strides[1] * self.stride_h,
            x.strides[2] * self.stride_w,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)
        out = windows.mean(axis=(3, 4))

        self._cache = {'out_h': out_h, 'out_w': out_w, 'x_shape': x.shape, 'single': single}
        return out[0] if single else out

    def backward(self, dout):
        c = self._cache
        single = c['single']
        if single:
            dout = dout[np.newaxis]

        N, H, W, C = c['x_shape']
        out_h, out_w = c['out_h'], c['out_w']
        dx = np.zeros((N, H, W, C), dtype=np.float32)
        scale = 1.0 / (self.pH * self.pW)

        for i in range(out_h):
            for j in range(out_w):
                h0, w0 = i * self.stride_h, j * self.stride_w
                dx[:, h0:h0+self.pH, w0:w0+self.pW, :] += scale * dout[:, i:i+1, j:j+1, :]

        return dx[0] if single else dx


class GlobalMaxPooling2DLayer:
    def __init__(self):
        self._cache = {}

    def forward(self, x):
        self._cache = {'x': np.asarray(x, dtype=np.float32)}
        if x.ndim == 3:
            return np.max(x, axis=(0, 1))
        return np.max(x, axis=(1, 2))

    def backward(self, dout):
        x = self._cache['x']
        single = x.ndim == 3
        if single:
            x   = x[np.newaxis]
            dout = dout[np.newaxis]

        max_vals = x.max(axis=(1, 2), keepdims=True)
        mask = (x == max_vals).astype(np.float32)
        mask /= mask.sum(axis=(1, 2), keepdims=True)
        dx = mask * dout[:, np.newaxis, np.newaxis, :]

        return dx[0] if single else dx


class GlobalAvgPooling2DLayer:
    def __init__(self):
        self._cache = {}

    def forward(self, x):
        self._cache = {'x_shape': np.asarray(x, dtype=np.float32).shape}
        if x.ndim == 3:
            return np.mean(x, axis=(0, 1))
        return np.mean(x, axis=(1, 2))

    def backward(self, dout):
        shape = self._cache['x_shape']
        single = len(shape) == 3

        if single:
            H, W, C = shape
            scale = 1.0 / (H * W)
            return np.ones((H, W, C), dtype=np.float32) * scale * dout[np.newaxis, np.newaxis, :]
        else:
            N, H, W, C = shape
            scale = 1.0 / (H * W)
            return np.ones((N, H, W, C), dtype=np.float32) * scale * dout[:, np.newaxis, np.newaxis, :]


class FlattenLayer:
    def __init__(self):
        self._cache = {}

    def forward(self, x):
        self._cache = {'shape': x.shape}
        if x.ndim == 3:
            return x.flatten(order="C")
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._cache['shape'])


class DenseLayer:
    def __init__(self, kernel, bias, activation=None):
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias,   dtype=np.float32)
        self._act_name = activation
        self.activation = get_activation(activation)
        self._cache = {}

    def forward(self, x):
        self._cache['x'] = x
        z = x @ self.kernel + self.bias
        self._cache['z'] = z
        out = self.activation(z) if self.activation else z
        self._cache['out'] = out
        return out

    def backward(self, dout):
        x = self._cache['x']
        dz = _act_backward(dout, self._act_name, self._cache['z'], self._cache['out'])

        single = x.ndim == 1
        if single:
            x = x[np.newaxis]
            dz = dz[np.newaxis]

        dx = dz @ self.kernel.T
        dW = x.T @ dz
        db = dz.sum(axis=0)

        return (dx[0] if single else dx), dW, db


class BatchNormLayer:
    def __init__(self, gamma, beta, moving_mean, moving_var, epsilon=1e-3):
        self.gamma = np.asarray(gamma,       dtype=np.float32)
        self.beta = np.asarray(beta,        dtype=np.float32)
        self.mean = np.asarray(moving_mean, dtype=np.float32)
        self.var = np.asarray(moving_var,  dtype=np.float32)
        self.eps = epsilon
        self._cache = {}

    def forward(self, x):
        sigma = np.sqrt(self.var + self.eps)
        x_hat = (x - self.mean) / sigma
        out = self.gamma * x_hat + self.beta
        self._cache = {'x_hat': x_hat, 'sigma': sigma}
        return out

    def backward(self, dout):
        x_hat = self._cache['x_hat']
        sigma = self._cache['sigma']

        d_beta = dout.sum(axis=0) if dout.ndim > 1 else dout
        d_gamma = (dout * x_hat).sum(axis=0) if dout.ndim > 1 else dout * x_hat
        dx = dout * self.gamma / sigma

        return dx, d_gamma, d_beta
