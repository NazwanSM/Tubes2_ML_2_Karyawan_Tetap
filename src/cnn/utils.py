import os
import numpy as np
from PIL import Image


def load_image(path, target_size=(150, 150)):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_batch(paths, target_size=(150, 150)):
    return np.stack([load_image(p, target_size) for p in paths], axis=0)


def extract_features(paths, keras_encoder, cache_path,
                    target_size=(299, 299), preprocess_fn=None, batch_size=32):
    
    if os.path.exists(cache_path):
        print(f"Nemu cache di {cache_path}, langsung load aja.")
        return np.load(cache_path)

    all_features = []
    n = len(paths)

    for start in range(0, n, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_batch(batch_paths, target_size)

        if preprocess_fn is not None:
            batch = preprocess_fn(batch)

        feats = keras_encoder.predict(batch, verbose=0)
        all_features.append(feats)

        done = min(start + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            print(f"  {done}/{n} gambar di-proses")

    features = np.concatenate(all_features, axis=0)
    np.save(cache_path, features)
    print(f"Features tersimpan -> {cache_path}  shape: {features.shape}")
    return features
