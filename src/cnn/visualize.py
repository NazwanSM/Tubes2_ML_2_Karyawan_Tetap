import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from src.cnn.layers import Conv2DLayer, LocallyConnected2DLayer


def _get_intermediate_outputs(scratch_model, image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]

    outputs = []
    out = image
    for layer in scratch_model.layers:
        out = layer.forward(out)
        outputs.append(out.copy())
    return outputs


def plot_feature_maps(scratch_model, image, max_filters=32, figsize_per=1.5):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]

    outputs = _get_intermediate_outputs(scratch_model, image)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title('Input Image')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    for i, layer in enumerate(scratch_model.layers):
        if not isinstance(layer, (Conv2DLayer, LocallyConnected2DLayer)):
            continue

        fmap = outputs[i] # (H, W, C)
        n_show = min(fmap.shape[-1], max_filters)
        n_cols = min(n_show, 8)
        n_rows = (n_show + n_cols - 1) // n_cols

        layer_type = type(layer).__name__.replace('Layer', '')
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * figsize_per, n_rows * figsize_per + 0.6)
        )
        axes = np.array(axes).flatten()
        fig.suptitle(
            f'Layer {i} — {layer_type}  |  output shape: {fmap.shape}',
            fontsize=10
        )

        for f in range(len(axes)):
            ax = axes[f]
            if f < n_show:
                ch = fmap[:, :, f]
                ax.imshow(ch, cmap='viridis')
                ax.set_title(f'f{f}', fontsize=7)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def grad_cam(keras_model, image, class_idx=None, last_conv_layer_name=None):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis]

    if last_conv_layer_name is None:
        for layer in reversed(keras_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("Tidak ada Conv2D layer di model.")

    grad_model = tf.keras.Model(
        inputs = keras_model.inputs,
        outputs = [
            keras_model.get_layer(last_conv_layer_name).output,
            keras_model.output,
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image)
        if class_idx is None:
            class_idx = int(np.argmax(preds[0]))
        score = preds[:, class_idx]

    grads = tape.gradient(score, conv_outputs) # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # (C,)

    conv_out = conv_outputs[0] # (H, W, C)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis] # (H, W, 1)
    heatmap = tf.squeeze(heatmap).numpy() # (H, W)
    heatmap = np.maximum(heatmap, 0) # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap, class_idx


def plot_grad_cam(keras_model, image, class_names=None, class_idx=None, last_conv_layer_name=None, alpha=0.45):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]

    heatmap, pred_cls = grad_cam(keras_model, image, class_idx, last_conv_layer_name)

    heatmap_rsz = tf.image.resize(
        heatmap[..., np.newaxis],
        (image.shape[0], image.shape[1])
    ).numpy().squeeze()

    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_rsz)[:, :, :3] # (H, W, 3)
    overlay = np.clip((1 - alpha) * image + alpha * heatmap_colored, 0, 1)

    label = class_names[pred_cls] if class_names else f'class {pred_cls}'

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap_rsz, cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap\n(pred: {label})')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(f'Grad-CAM — Predicted: {label}', fontsize=11)
    plt.tight_layout()
    plt.show()

    return heatmap, pred_cls


def plot_grad_cam_grid(keras_model, images, class_names=None, last_conv_layer_name=None, alpha=0.45, n_cols=4):
    images = [np.asarray(img, dtype=np.float32) for img in images]
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 2,
                              figsize=(n_cols * 2 * 3, n_rows * 3.2))
    axes = np.array(axes).reshape(n_rows, n_cols * 2)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = (idx % n_cols) * 2

        heatmap, pred_cls = grad_cam(keras_model, img, None, last_conv_layer_name)

        heatmap_rsz = tf.image.resize(
            heatmap[..., np.newaxis],
            (img.shape[0], img.shape[1])
        ).numpy().squeeze()

        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_rsz)[:, :, :3]
        overlay = np.clip((1 - alpha) * img + alpha * heatmap_colored, 0, 1)

        label = class_names[pred_cls] if class_names else f'cls {pred_cls}'

        axes[row, col].imshow(np.clip(img, 0, 1))
        axes[row, col].set_title(f'pred: {label}', fontsize=8)
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].set_title('Grad-CAM', fontsize=8)
        axes[row, col + 1].axis('off')

    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    plt.suptitle('Grad-CAM Grid', fontsize=12)
    plt.tight_layout()
    plt.show()
