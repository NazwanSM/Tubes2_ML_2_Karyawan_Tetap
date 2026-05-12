import math
import numpy as np

#  Fungsi Aktivasi
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


#  Fungsi Helper
def _same_pad(H, W, kH, kW, stride_h, stride_w):
    out_h = math.ceil(H / stride_h)
    out_w = math.ceil(W / stride_w)
    pad_h = max((out_h - 1) * stride_h + kH - H, 0)
    pad_w = max((out_w - 1) * stride_w + kW - W, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


#  Layer Implementations  
class Conv2DLayer:
    def __init__(self, kernel, bias, strides=(1, 1), padding="valid", activation=None):
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias   = np.asarray(bias,   dtype=np.float32)
        self.stride_h, self.stride_w = strides
        self.padding   = padding.lower()
        self.activation = get_activation(activation)
        self.kH, self.kW = self.kernel.shape[:2]

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        H, W, C = x.shape

        if self.padding == "same":
            pt, pb, pl, pr = _same_pad(H, W, self.kH, self.kW, self.stride_h, self.stride_w)
            x = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
            H, W = x.shape[:2]

        out_h = (H - self.kH) // self.stride_h + 1
        out_w = (W - self.kW) // self.stride_w + 1
        C_out = self.kernel.shape[-1]

        patch_shape   = (out_h, out_w, self.kH, self.kW, C)
        patch_strides = (
            x.strides[0] * self.stride_h,
            x.strides[1] * self.stride_w,
            x.strides[0],
            x.strides[1],
            x.strides[2],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=patch_shape, strides=patch_strides)

        cols   = patches.reshape(out_h * out_w, -1)       # (out_h*out_w, kH*kW*C_in)
        W_flat = self.kernel.reshape(-1, C_out)            # (kH*kW*C_in, C_out)
        out    = (cols @ W_flat + self.bias).reshape(out_h, out_w, C_out)

        return self.activation(out) if self.activation else out


class LocallyConnected2DLayer:
    def __init__(self, kernel, bias, kernel_size, strides=(1, 1), padding="valid", activation=None):
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias   = np.asarray(bias,   dtype=np.float32)
        ks = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.kH, self.kW = ks
        self.stride_h, self.stride_w = strides
        self.padding    = padding.lower()
        self.activation = get_activation(activation)

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        H, W, C = x.shape

        if self.padding == "same":
            pt, pb, pl, pr = _same_pad(H, W, self.kH, self.kW, self.stride_h, self.stride_w)
            x = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
            H, W = x.shape[:2]

        out_h = (H - self.kH) // self.stride_h + 1
        out_w = (W - self.kW) // self.stride_w + 1
        C_out = self.kernel.shape[-1]
        out   = np.zeros((out_h, out_w, C_out), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[
                    i * self.stride_h : i * self.stride_h + self.kH,
                    j * self.stride_w : j * self.stride_w + self.kW,
                    :,
                ]
                # kernel[i, j] shape: (kH*kW*C_in, C_out)
                out[i, j] = patch.flatten() @ self.kernel[i, j] + self.bias[i, j]

        return self.activation(out) if self.activation else out


class MaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pH, self.pW = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        H, W, C = x.shape
        out_h = (H - self.pH) // self.stride_h + 1
        out_w = (W - self.pW) // self.stride_w + 1

        win_shape   = (out_h, out_w, self.pH, self.pW, C)
        win_strides = (
            x.strides[0] * self.stride_h,
            x.strides[1] * self.stride_w,
            x.strides[0],
            x.strides[1],
            x.strides[2],
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)
        return windows.max(axis=(2, 3))   # (out_h, out_w, C)


class AvgPooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pH, self.pW = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        H, W, C = x.shape
        out_h = (H - self.pH) // self.stride_h + 1
        out_w = (W - self.pW) // self.stride_w + 1

        win_shape   = (out_h, out_w, self.pH, self.pW, C)
        win_strides = (
            x.strides[0] * self.stride_h,
            x.strides[1] * self.stride_w,
            x.strides[0],
            x.strides[1],
            x.strides[2],
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)
        return windows.mean(axis=(2, 3))  # (out_h, out_w, C)


class GlobalMaxPooling2DLayer:
    def forward(self, x):
        return np.max(x, axis=(0, 1))   # (H, W, C) -> (C,)


class GlobalAvgPooling2DLayer:
    def forward(self, x):
        return np.mean(x, axis=(0, 1))  # (H, W, C) -> (C,)


class FlattenLayer:
    def forward(self, x):
        return x.flatten(order="C")


class DenseLayer:
    def __init__(self, kernel, bias, activation=None):
        # kernel dari Keras: (in_dim, out_dim)
        self.kernel     = np.asarray(kernel, dtype=np.float32)
        self.bias       = np.asarray(bias,   dtype=np.float32)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = x @ self.kernel + self.bias
        return self.activation(out) if self.activation else out


class BatchNormLayer:
    def __init__(self, gamma, beta, moving_mean, moving_var, epsilon=1e-3):
        self.gamma = np.asarray(gamma,       dtype=np.float32)
        self.beta  = np.asarray(beta,        dtype=np.float32)
        self.mean  = np.asarray(moving_mean, dtype=np.float32)
        self.var   = np.asarray(moving_var,  dtype=np.float32)
        self.eps   = epsilon

    def forward(self, x):
        x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * x_norm + self.beta
