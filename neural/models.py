from collections import Counter
import copy
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleTokenizer:
    def __init__(self, num_words=10000, char_level=False, filters="", oov_token="oov", lower=True):
        self.num_words = num_words
        self.char_level = char_level
        self.filters = filters
        self.oov_token = oov_token
        self.lower = lower
        self.pad_token = "<pad>"
        self.pad_index = 0
        self.oov_index = 1
        self.word_index = {self.pad_token: self.pad_index, self.oov_token: self.oov_index}

    def _normalize(self, text):
        if isinstance(text, list):
            tokens = [token.lower() if self.lower and isinstance(token, str) else token for token in text]
        else:
            normalized = text.lower() if self.lower and isinstance(text, str) else text
            if self.char_level:
                tokens = list(normalized)
            else:
                tokens = normalized.split()
        return tokens

    def fit_on_texts(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._normalize(text))
        limit = None if self.num_words is None else max(0, self.num_words - 2)
        most_common = counter.most_common(limit)
        self.word_index = {self.pad_token: self.pad_index, self.oov_token: self.oov_index}
        for index, (token, _) in enumerate(most_common, start=2):
            self.word_index[token] = index

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = self._normalize(text)
            sequences.append([self.word_index.get(token, self.oov_index) for token in tokens])
        return sequences

    @property
    def vocab_size(self):
        return len(self.word_index)


class TorchRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, stacks, output_mlp_size, use_gru=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_layers = max(1, stacks + 1)
        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_mlp_size),
            nn.ReLU(),
            nn.Linear(output_mlp_size, vocab_size),
        )

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        encoded, _ = self.rnn(embedded)
        return self.output(encoded[:, -1, :])


def get_plato_rnn():
    from urllib.request import urlopen

    rnn_lm = RNN(epochs=1000, patience=10)
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato[:10000])
    return rnn_lm


def load(model_path="rnn"):
    checkpoint = torch.load(model_path + ".pt", map_location="cpu")
    rnn = RNN(**checkpoint["config"])
    with open(model_path + ".tkn", "rb") as handle:
        rnn.tokenizer = pickle.load(handle)
    rnn.set_up_indices()
    rnn.build(vocab_size=checkpoint["vocab_size"])
    rnn.model.load_state_dict(checkpoint["state_dict"])
    rnn.model.eval()
    return rnn


class GPT2:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate_next_gram(self, context="", N=3):
        input_ids = self.tokenizer.encode(context, return_tensors="pt")
        output_ids = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=N,
            max_length=input_ids.shape[1] + 1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        words = [self.tokenizer.decode(w[-1:], skip_special_tokens=True).strip() for w in output_ids]
        return words


class RNN:
    """
    from neural import models
    rnn_lm = models.RNN()
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato)
    """

    def __init__(
        self,
        stacks=0,
        split=0.1,
        vocab_size=10000,
        batch_size=128,
        epochs=100,
        patience=3,
        hidden_size=50,
        window=3,
        char_level=False,
        max_steps=10000000,
        use_gru=False,
        monitor="val_loss",
        mode="min",
        lower=True,
        oov="oov",
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.oov = oov
        self.lower = lower
        self.hidden_size = hidden_size
        self.output_mlp_size = 100
        self.embedding_size = 200
        self.use_gru = use_gru
        self.name = "rnn"
        self.char_level = char_level
        self.window = window
        self.max_steps = max_steps
        self.stacks = stacks
        self.vocab_size = vocab_size
        self.split = split
        self.monitor = monitor
        self.mode = mode
        self.tokenizer = None
        self.i2w = None
        self.w2i = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 2 * self.window - 1

    def build(self, vocab_size=None):
        model_vocab_size = vocab_size or self.vocab_size
        self.model = TorchRNNModel(
            vocab_size=model_vocab_size,
            embedding_dim=self.embedding_size,
            hidden_size=self.hidden_size,
            stacks=self.stacks,
            output_mlp_size=self.output_mlp_size,
            use_gru=self.use_gru,
        )
        self.model.to(self.device)

    def set_up_indices(self):
        self.i2w = {index: word for word, index in self.tokenizer.word_index.items()}
        self.w2i = dict(self.tokenizer.word_index)

    def text_to_sequences(self, text):
        self.tokenizer = SimpleTokenizer(
            num_words=self.vocab_size,
            char_level=self.char_level,
            filters="",
            oov_token=self.oov,
            lower=self.lower,
        )
        self.tokenizer.fit_on_texts([text])
        self.set_up_indices()
        print("Vocabulary Size: %d" % self.tokenizer.vocab_size)
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        windows = list(range(self.window, len(encoded) - self.window))
        sequences = np.array([np.zeros(self.window * 2, dtype=np.int64) for _ in windows], dtype=np.int64)
        for i, w in enumerate(windows):
            sequences[i] = np.array(encoded[w - self.window : w + self.window], dtype=np.int64)
        print("Total Sequences: %d" % len(sequences))
        if len(sequences) == 0:
            return np.empty((0, self.sequence_length), dtype=np.int64), np.empty((0,), dtype=np.int64)
        x = sequences[:, :-1]
        y = sequences[:, -1]
        return x, y

    def _prepare_context(self, history):
        encoded = self.tokenizer.texts_to_sequences([history])[0]
        context = encoded[-self.sequence_length :]
        if len(context) < self.sequence_length:
            context = [self.tokenizer.pad_index] * (self.sequence_length - len(context)) + context
        return np.array([context], dtype=np.int64)

    def _predict_logits(self, contexts):
        if len(contexts) == 0:
            vocab_size = len(self.i2w) if self.i2w is not None else self.vocab_size
            return np.empty((0, vocab_size))
        tensor = torch.as_tensor(contexts, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def _predict_top_indices(self, context, top_n=1):
        scores = self._predict_logits(context)[0]
        return scores.argsort()[-top_n:][::-1], scores

    def train(self, text):
        x, y = self.text_to_sequences(text)
        if len(x) == 0:
            raise ValueError("Not enough tokens to build training windows for the configured window size.")
        self.build(vocab_size=self.tokenizer.vocab_size)
        x_tensor = torch.as_tensor(x, dtype=torch.long)
        y_tensor = torch.as_tensor(y, dtype=torch.long)

        split_index = int(len(x_tensor) * (1 - self.split)) if self.split > 0 else len(x_tensor)
        if split_index <= 0 or split_index >= len(x_tensor):
            train_x, train_y = x_tensor, y_tensor
            val_x, val_y = None, None
        else:
            train_x, train_y = x_tensor[:split_index], y_tensor[:split_index]
            val_x, val_y = x_tensor[split_index:], y_tensor[split_index:]

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=min(self.batch_size, len(train_x)), shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_metric = np.inf if self.mode == "min" else -np.inf
        patience_left = self.patience

        for _ in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if val_x is None or len(val_x) == 0:
                best_state = copy.deepcopy(self.model.state_dict())
                continue

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_x.to(self.device))
                val_loss = criterion(val_logits, val_y.to(self.device)).item()

            improved = val_loss < best_metric if self.mode == "min" else val_loss > best_metric
            if improved:
                best_metric = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def generate_next_gram(self, history, top_n=1):
        """
        Return the next gram (character/word) given a preceding text.
        When top_n>1, more suggestions are returned.
        """
        context_encoded = self._prepare_context(history)
        top_indices, _ = self._predict_top_indices(context_encoded, top_n=top_n)
        if top_n == 1:
            return self.i2w.get(int(top_indices[0]), self.oov)
        return [self.i2w.get(int(index), self.oov) for index in top_indices]

    def generate_seq(self, seed_text, n_words):
        out_text = seed_text
        separator = "" if self.char_level else " "
        for _ in range(n_words):
            out_word = self.generate_next_gram(out_text)
            out_text += separator + out_word
        return out_text

    def compute_gram_probs(self, text):
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = self.sequence_length
        probs = []
        for i in range(history, len(encoded)):
            target = encoded[i]
            context_encoded = np.array([encoded[i - history : i]], dtype=np.int64)
            p = self._predict_logits(context_encoded)[0][target]
            probs.append(p)
        return probs

    def predict_words(self, text):
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = self.sequence_length
        context_windows = [encoded[i - history : i] for i in range(history, len(encoded))]
        prediction_scores = self._predict_logits(np.array(context_windows, dtype=np.int64))
        predicted_indices = np.argmax(prediction_scores, axis=1)
        return [None for _ in range(history)] + [self.i2w.get(int(i), self.oov) for i in predicted_indices]

    def cross_entropy(self, text, PPL=False):
        probs = self.compute_gram_probs(text)
        log_probs = list(map(np.log2, probs))
        ce = -np.mean(log_probs)
        return np.power(2, ce) if PPL else ce

    def save(self, name="rnn"):
        checkpoint = {
            "config": {
                "stacks": self.stacks,
                "split": self.split,
                "vocab_size": self.vocab_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "patience": self.patience,
                "hidden_size": self.hidden_size,
                "window": self.window,
                "char_level": self.char_level,
                "max_steps": self.max_steps,
                "use_gru": self.use_gru,
                "monitor": self.monitor,
                "mode": self.mode,
                "lower": self.lower,
                "oov": self.oov,
            },
            "vocab_size": self.tokenizer.vocab_size,
            "state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, f"{name}.pt")
        with open(f"{name}.tkn", "wb") as handle:
            pickle.dump(self.tokenizer, handle)

    def accuracy(self, text, unwanted_term="xxxx", oov="oov", lexicon={}, relative_kd=True):
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = self.sequence_length
        scores = []
        keystrokes, keystrokes_discounted = 0, 0
        for i in range(history, len(encoded)):
            target = encoded[i]
            target_word = self.i2w[target]
            if unwanted_term in target_word:
                continue
            if target_word == oov or (len(lexicon) > 0 and (target_word not in lexicon)):
                scores.append(0)
                keystrokes += len(target_word)
                keystrokes_discounted += len(target_word)
                continue
            context_encoded = np.array([encoded[i - history : i]], dtype=np.int64)
            predicted = int(np.argmax(self._predict_logits(context_encoded), axis=-1)[0])
            if target == predicted:
                scores.append(1)
                keystrokes += len(target_word)
                keystrokes_discounted += 1
            else:
                scores.append(0)
                keystrokes += len(target_word)
                keystrokes_discounted += len(target_word)
        return np.mean(scores), 1 - (keystrokes_discounted / keystrokes) if relative_kd else (keystrokes_discounted, keystrokes)
