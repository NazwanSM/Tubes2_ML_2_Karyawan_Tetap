import numpy as np

from src.rnn_lstm.layers import EmbeddingLayer, SimpleRNNCell, LSTMCell, DenseLayer
from src.cnn.utils import load_image


class CaptioningFromScratch:

    def __init__(self, cnn_encoder, embedding, proj_dense, rnn_cells, output_dense,
                 img_size=(299, 299), preprocess_fn=None):
        self.cnn_encoder  = cnn_encoder   
        self.embedding    = embedding      
        self.proj_dense   = proj_dense     
        self.rnn_cells    = rnn_cells     
        self.output_dense = output_dense   
        self.img_size     = img_size
        self.preprocess_fn = preprocess_fn  
        self.is_lstm      = isinstance(rnn_cells[0], LSTMCell)

    @classmethod
    def from_keras(cls, cnn_encoder, decoder_model, img_size=(299, 299), preprocess_fn=None):
        by_type = {}
        for layer in decoder_model.layers:
            key = type(layer).__name__
            by_type.setdefault(key, []).append(layer)

        # Embedding layer
        emb_layer = by_type["Embedding"][0]
        embedding = EmbeddingLayer(emb_layer.get_weights()[0])
        embed_dim = embedding.embed_dim

        # Dense layers 
        dense_layers = by_type.get("Dense", [])
        assert len(dense_layers) >= 2, "Minimal butuh 2 Dense: projection + output"

        proj_keras  = min(dense_layers, key=lambda l: l.output_shape[-1])
        out_keras   = max(dense_layers, key=lambda l: l.output_shape[-1])

        proj_k, proj_b = proj_keras.get_weights()
        proj_dense = DenseLayer(proj_k, proj_b, activation=None)

        out_k, out_b = out_keras.get_weights()
        output_dense = DenseLayer(out_k, out_b, activation="softmax")

        # RNN / LSTM layers 
        rnn_cells = []
        rnn_type  = None

        for cell_cls, name in [(LSTMCell, "LSTM"), (SimpleRNNCell, "SimpleRNN")]:
            if name in by_type:
                rnn_type = name
                for rnn_layer in by_type[name]:
                    w = rnn_layer.get_weights() 
                    rnn_cells.append(cell_cls(w[0], w[1], w[2]))
                break

        assert rnn_cells, "Ga nemu SimpleRNN atau LSTM di decoder model"
        print(f"Loaded decoder: {rnn_type} x{len(rnn_cells)} layer(s)")

        return cls(cnn_encoder, embedding, proj_dense, rnn_cells, output_dense,
                   img_size=img_size, preprocess_fn=preprocess_fn)

    # helpers
    def _extract_feature(self, img_path):
        img = load_image(img_path, target_size=self.img_size)
        batch = img[np.newaxis, ...]  
        if self.preprocess_fn:
            batch = self.preprocess_fn(batch)
        return self.cnn_encoder.predict(batch, verbose=0)[0]  

    def _init_states(self):
        if self.is_lstm:
            return [cell.zero_state() for cell in self.rnn_cells] 
        return [cell.zero_state() for cell in self.rnn_cells]      

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

    def _decode(self, cnn_feature, word2idx, idx2word, max_len=50):
        start_id = word2idx["<start>"]
        end_id   = word2idx["<end>"]
        img_proj = self.proj_dense.forward(cnn_feature) 
        states = self._init_states()
        
        _, states = self._step(img_proj, states)
        
        current_token = start_id
        caption_ids   = []

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

    # interface 
    def generate_caption(self, img_path, word2idx, idx2word, max_len=50):
        feature = self._extract_feature(img_path)
        return self._decode(feature, word2idx, idx2word, max_len)

    def generate_from_feature(self, cnn_feature, word2idx, idx2word, max_len=50):
        return self._decode(cnn_feature, word2idx, idx2word, max_len)

    def generate_batch(self, cnn_features, word2idx, idx2word, max_len=50):
        return [
            self.generate_from_feature(feat, word2idx, idx2word, max_len)
            for feat in cnn_features
        ]
