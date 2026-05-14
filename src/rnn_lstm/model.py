import numpy as np

from src.rnn_lstm.layers import EmbeddingLayer, SimpleRNNCell, LSTMCell, DenseLayer
from src.cnn.utils import load_image


class CaptioningFromScratch:

    def __init__(self, cnn_encoder, embedding, proj_dense, rnn_cells, output_dense, arch='inject', img_size=(299, 299), preprocess_fn=None):
        self.cnn_encoder = cnn_encoder
        self.embedding = embedding
        self.proj_dense = proj_dense
        self.rnn_cells = rnn_cells
        self.output_dense = output_dense
        self.arch = arch          
        self.img_size = img_size
        self.preprocess_fn = preprocess_fn
        self.is_lstm = isinstance(rnn_cells[0], LSTMCell)

    @classmethod
    def from_keras(cls, cnn_encoder, decoder_model, img_size=(299, 299), preprocess_fn=None):
        by_type = {}
        for layer in decoder_model.layers:
            key = type(layer).__name__
            by_type.setdefault(key, []).append(layer)

        arch = 'init_inject' if 'Add' in by_type else 'inject'

        # Embedding
        emb_layer = by_type["Embedding"][0]
        embedding = EmbeddingLayer(emb_layer.get_weights()[0])

        # Dense layers 
        dense_layers = by_type.get("Dense", [])
        assert len(dense_layers) >= 2, "Minimal butuh 2 Dense: projection + output"

        proj_keras = min(dense_layers, key=lambda l: l.output_shape[-1])
        out_keras = max(dense_layers, key=lambda l: l.output_shape[-1])

        proj_k, proj_b = proj_keras.get_weights()
        proj_dense = DenseLayer(proj_k, proj_b, activation=None)

        out_k, out_b = out_keras.get_weights()
        output_dense = DenseLayer(out_k, out_b, activation="softmax")

        # RNN / LSTM cells
        rnn_cells = []
        rnn_type = None
        for cell_cls, name in [(LSTMCell, "LSTM"), (SimpleRNNCell, "SimpleRNN")]:
            if name in by_type:
                rnn_type = name
                for rnn_layer in by_type[name]:
                    w = rnn_layer.get_weights()
                    rnn_cells.append(cell_cls(w[0], w[1], w[2]))
                break

        assert rnn_cells, "Ga nemu SimpleRNN atau LSTM di decoder model"
        print(f"Loaded decoder ({arch}): {rnn_type} x{len(rnn_cells)} layer(s)")

        return cls(cnn_encoder, embedding, proj_dense, rnn_cells, output_dense,
                   arch=arch, img_size=img_size, preprocess_fn=preprocess_fn)

    # helpers
    def _extract_feature(self, img_path):
        img = load_image(img_path, target_size=self.img_size)
        batch = img[np.newaxis, ...]
        if self.preprocess_fn:
            batch = self.preprocess_fn(batch)
        return self.cnn_encoder.predict(batch, verbose=0)[0]

    def _init_states(self, batch_size=1):
        return [cell.zero_state(batch_size) for cell in self.rnn_cells]

    def _step(self, x, states):
        new_states = []
        for l, cell in enumerate(self.rnn_cells):
            if self.is_lstm:
                h, c = states[l]
                h, c = cell.forward(x, h, c)
                new_states.append((h, c))
                x = h
            else:
                h = cell.forward(x, states[l])
                new_states.append(h)
                x = h
        return x, new_states

    # single decode
    def _decode(self, cnn_feature, word2idx, idx2word, max_len=50):
        start_id = word2idx["<start>"]
        end_id = word2idx["<end>"]

        img_proj = self.proj_dense.forward(cnn_feature)
        states = self._init_states(batch_size=1)
        _, states = self._step(img_proj, states)

        current_token = start_id
        caption_ids = []
        for _ in range(max_len):
            x = self.embedding.forward(current_token)
            x, states = self._step(x, states)
            logits = self.output_dense.forward(x)
            next_token = int(np.argmax(logits))
            if next_token == end_id:
                break
            caption_ids.append(next_token)
            current_token = next_token

        return " ".join(idx2word.get(i, "<unk>") for i in caption_ids)

    def _decode_init_inject(self, cnn_feature, word2idx, idx2word, max_len=50):
        start_id = word2idx["<start>"]
        end_id = word2idx["<end>"]

        img_proj = self.proj_dense.forward(cnn_feature)   # (hidden_size,)
        states = self._init_states(batch_size=1)

        current_token = start_id
        caption_ids = []
        for _ in range(max_len):
            x = self.embedding.forward(current_token) # (embed_dim,)
            x, states = self._step(x, states) # (hidden_size,)
            x_combined = x + img_proj # Add context
            logits = self.output_dense.forward(x_combined)
            next_token = int(np.argmax(logits))
            if next_token == end_id:
                break
            caption_ids.append(next_token)
            current_token = next_token

        return " ".join(idx2word.get(i, "<unk>") for i in caption_ids)

    # batch decode
    def _decode_batch(self, cnn_features, word2idx, idx2word, max_len=50):
        N = len(cnn_features)
        start_id = word2idx["<start>"]
        end_id = word2idx["<end>"]

        img_proj = self.proj_dense.forward(
            np.array(cnn_features, dtype=np.float32)
        )
        states = self._init_states(batch_size=N)
        _, states = self._step(img_proj, states)

        current_tokens = np.full(N, start_id, dtype=np.int32)
        caption_ids = [[] for _ in range(N)]
        done = np.zeros(N, dtype=bool)

        for _ in range(max_len):
            x = self.embedding.forward_batch(current_tokens)
            x, states = self._step(x, states)
            logits = self.output_dense.forward(x)
            next_tokens = logits.argmax(axis=-1)

            for i in range(N):
                if not done[i]:
                    if next_tokens[i] == end_id:
                        done[i] = True
                    else:
                        caption_ids[i].append(int(next_tokens[i]))

            current_tokens = next_tokens
            if done.all():
                break

        return [" ".join(idx2word.get(i, "<unk>") for i in ids) for ids in caption_ids]

    def _decode_batch_init_inject(self, cnn_features, word2idx, idx2word, max_len=50):
        N = len(cnn_features)
        start_id = word2idx["<start>"]
        end_id = word2idx["<end>"]

        img_proj = self.proj_dense.forward(
            np.array(cnn_features, dtype=np.float32) # (N, hidden_size)
        )                                                   
        states = self._init_states(batch_size=N)

        current_tokens = np.full(N, start_id, dtype=np.int32)
        caption_ids = [[] for _ in range(N)]
        done = np.zeros(N, dtype=bool)

        for _ in range(max_len):
            x = self.embedding.forward_batch(current_tokens) # (N, embed_dim)
            x, states = self._step(x, states) # (N, hidden_size)
            x_combined = x + img_proj # Add context
            logits = self.output_dense.forward(x_combined)
            next_tokens = logits.argmax(axis=-1)

            for i in range(N):
                if not done[i]:
                    if next_tokens[i] == end_id:
                        done[i] = True
                    else:
                        caption_ids[i].append(int(next_tokens[i]))

            current_tokens = next_tokens
            if done.all():
                break

        return [" ".join(idx2word.get(i, "<unk>") for i in ids) for ids in caption_ids]

    # interface 
    def generate_caption(self, img_path, word2idx, idx2word, max_len=50):
        feature = self._extract_feature(img_path)
        return self.generate_from_feature(feature, word2idx, idx2word, max_len)

    def generate_from_feature(self, cnn_feature, word2idx, idx2word, max_len=50):
        if self.arch == 'init_inject':
            return self._decode_init_inject(cnn_feature, word2idx, idx2word, max_len)
        return self._decode(cnn_feature, word2idx, idx2word, max_len)

    def generate_batch(self, cnn_features, word2idx, idx2word, max_len=50, batch_size=32):
        cnn_features = np.asarray(cnn_features, dtype=np.float32)
        decode_fn = (self._decode_batch_init_inject
                        if self.arch == 'init_inject'
                        else self._decode_batch)
        results = []
        for i in range(0, len(cnn_features), batch_size):
            batch = cnn_features[i:i + batch_size]
            results.extend(decode_fn(batch, word2idx, idx2word, max_len))
        return results
