import re
import json
import numpy as np
import pandas as pd

SPECIAL_TOKENS = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}


def load_captions(csv_path):
    df = pd.read_csv(csv_path)
    captions = {}
    for _, row in df.iterrows():
        img = row["image"]
        cap = row["caption"]
        captions.setdefault(img, []).append(cap)
    return captions


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(split_txt_path):
    with open(split_txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def auto_split(captions_dict, n_train=6000, n_val=1000, n_test=1000, seed=42):
    all_images = list(captions_dict.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_images)
    train = all_images[:n_train]
    val   = all_images[n_train : n_train + n_val]
    test  = all_images[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def build_vocab(captions_dict, train_images, min_freq=1):
    freq = {}
    for img in train_images:
        for cap in captions_dict.get(img, []):
            for word in clean_text(cap).split():
                freq[word] = freq.get(word, 0) + 1

    word2idx = dict(SPECIAL_TOKENS)  
    for word in sorted(freq):
        if freq[word] >= min_freq and word not in word2idx:
            word2idx[word] = len(word2idx)

    print(f"Vocab size: {len(word2idx)} kata (min_freq={min_freq})")
    return word2idx


def save_vocab(word2idx, path):
    with open(path, "w") as f:
        json.dump(word2idx, f, indent=2)


def load_vocab(path):
    with open(path) as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def _tokenize_one(caption, word2idx, max_len):
    pad_id   = word2idx["<pad>"]
    start_id = word2idx["<start>"]
    end_id   = word2idx["<end>"]
    unk_id   = word2idx["<unk>"]

    words = clean_text(caption).split()[: max_len - 2]  
    tokens = [start_id] + [word2idx.get(w, unk_id) for w in words] + [end_id]

    tokens += [pad_id] * (max_len + 1 - len(tokens))
    return tokens[: max_len + 1]


def make_sequences(captions_dict, image_names, word2idx, max_len):
    img_list, cap_in_list, cap_target_list = [], [], []

    for img in image_names:
        for cap in captions_dict.get(img, []):
            tokens = _tokenize_one(cap, word2idx, max_len)  
            cap_in_list.append(tokens[:max_len])           
            cap_target_list.append(tokens[1: max_len + 1]) 
            img_list.append(img)

    return img_list, np.array(cap_in_list, dtype=np.int32), np.array(cap_target_list, dtype=np.int32)
