import numpy as np

from src.cnn.layers import (
    Conv2DLayer,
    LocallyConnected2DLayer,
    MaxPooling2DLayer,
    AvgPooling2DLayer,
    GlobalMaxPooling2DLayer,
    GlobalAvgPooling2DLayer,
    FlattenLayer,
    DenseLayer,
    BatchNormLayer,
)

_SKIP_LAYERS = {"InputLayer", "Dropout", "Rescaling", "RandomFlip",
                "RandomRotation", "RandomZoom"}


def _convert_layer(keras_layer):
    layer_type = type(keras_layer).__name__

    if layer_type in _SKIP_LAYERS:
        return None

    cfg = keras_layer.get_config()
    weights = keras_layer.get_weights()

    if layer_type == "Conv2D":
        kernel, bias = weights
        return Conv2DLayer(
            kernel = kernel,
            bias = bias,
            strides = tuple(cfg["strides"]),
            padding = cfg["padding"],
            activation = cfg["activation"],
        )

    if layer_type == "LocallyConnected2D":
        kernel, bias = weights
        _, out_h, out_w, C_out = keras_layer.output_shape
        kernel = kernel.reshape(out_h, out_w, -1, C_out)
        bias   = bias.reshape(out_h, out_w, C_out)
        return LocallyConnected2DLayer(
            kernel = kernel,
            bias = bias,
            kernel_size = tuple(cfg["kernel_size"]),
            strides = tuple(cfg["strides"]),
            padding = cfg.get("padding", "valid"),
            activation = cfg["activation"],
        )

    if layer_type == "MaxPooling2D":
        strides = tuple(cfg["strides"]) if cfg.get("strides") else None
        return MaxPooling2DLayer(pool_size=tuple(cfg["pool_size"]), strides=strides)

    if layer_type == "AveragePooling2D":
        strides = tuple(cfg["strides"]) if cfg.get("strides") else None
        return AvgPooling2DLayer(pool_size=tuple(cfg["pool_size"]), strides=strides)

    if layer_type == "GlobalMaxPooling2D":
        return GlobalMaxPooling2DLayer()

    if layer_type == "GlobalAveragePooling2D":
        return GlobalAvgPooling2DLayer()

    if layer_type == "Flatten":
        return FlattenLayer()

    if layer_type == "Dense":
        kernel, bias = weights
        return DenseLayer(kernel=kernel, bias=bias, activation=cfg["activation"])

    if layer_type == "BatchNormalization":
        gamma, beta, moving_mean, moving_var = weights
        return BatchNormLayer(gamma, beta, moving_mean, moving_var, epsilon=cfg["epsilon"])

    print(f"[SKIP] Layer '{layer_type}' belum di-handle, di-lewatin.")
    return None


class CNNFromScratch:
    def __init__(self):
        self.layers = []

    @classmethod
    def from_keras(cls, keras_model):
        model = cls()
        skipped = []

        for layer in keras_model.layers:
            scratch_layer = _convert_layer(layer)
            if scratch_layer is not None:
                model.layers.append(scratch_layer)
            else:
                skipped.append(type(layer).__name__)

        print(f"Model loaded: {len(model.layers)} layer aktif, {len(skipped)} di-skip {skipped}")
        return model

    def predict(self, x, batch_size=32):
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        results = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size]
            out = batch
            for layer in self.layers:
                out = layer.forward(out)
            results.append(out)

        out = np.concatenate(results, axis=0)
        return out[0] if single else out

    def predict_classes(self, X, batch_size=32):
        return self.predict(X, batch_size=batch_size).argmax(axis=1)
