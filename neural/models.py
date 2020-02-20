import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

def demo():
    from urllib.request import urlopen
    rnn_lm = RNN()
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato[:1000])

class RNN:
    """
    from neural import models
    rnn_lm = models.RNN()
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato)
    """
    def __init__(self, vocab_size=10000, batch_size=128, epochs=100, patience=3, hidden_size=50, max_seq_len=512, window=3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.output_mlp_size = 100
        self.window = window
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.tokenizer = None
        self.i2w = None

    def build(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 200, input_length=2*self.window-1))
        self.model.add(LSTM(self.hidden_size, return_sequences=True))
        self.model.add(LSTM(self.hidden_size))
        self.model.add(Dense(self.output_mlp_size, activation='relu'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, text):
        x, y = self.text_to_sequences(text)
        self.build()
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=[self.early_stop])

    def text_to_sequences(self, text):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.i2w = {index: word for word, index in self.tokenizer.word_index.items()}
        print('Vocabulary Size: %d' % self.vocab_size)
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        sequences = list()
        # create equally-sized windows
        for i in range(self.window, len(encoded)-self.window):
            sequence = encoded[i - self.window:i + self.window]
            sequences.append(np.array(sequence))
        print('Total Sequences: %d' % len(sequences))
        sequences = np.array(sequences)
        # let the last token from each window be the target
        X, y = sequences[:,:-1], sequences[:,-1]
        # turn y to onehot
        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y

    # generate a sequence from the model
    def generate_seq(self, seed_text, n_words):
        out_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([out_text])[0]
            encoded = encoded[- 2 * self.window + 1:]
            # predict a word in the vocabulary
            predicted_index = self.model.predict_classes([encoded], verbose=0)
            # map predicted word index to word
            out_word = self.i2w[predicted_index[0]]
            # append to input
            out_text += " " + out_word
        return out_text