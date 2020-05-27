import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding


def get_plato_rnn():
    from urllib.request import urlopen
    rnn_lm = RNN(epochs=1000, patience=10)
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato[:10000])
    return rnn_lm

class RNN:
    """
    from neural import models
    rnn_lm = models.RNN()
    plato = urlopen("http://www.gutenberg.org/cache/epub/1497/pg1497.txt").read().decode("utf8")
    rnn_lm.train(plato)
    """
    def __init__(self, stacks=0, split=0.1, vocab_size=10000, batch_size=128, epochs=100, patience=3, hidden_size=50,
                 window=3, max_steps=10000000, use_gru=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.output_mlp_size = 100
        self.use_gru = use_gru
        self.name = "rnn"
        self.window = window
        self.max_steps = max_steps
        self.stacks = stacks
        self.vocab_size = vocab_size
        self.split = split
        self.early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.tokenizer = None
        self.i2w = None

    def build(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 200, input_length=2*self.window-1))
        RnnCell = GRU if self.use_gru else LSTM
        for stack in range(self.stacks):
            self.model.add(RnnCell(self.hidden_size, return_sequences=True))
        self.model.add(RnnCell(self.hidden_size))
        self.model.add(Dense(self.output_mlp_size, activation='relu'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, text):
        x, y = self.text_to_sequences(text)
        self.build()
        self.model.fit(x, y, validation_split=self.split, batch_size=self.batch_size, epochs=self.epochs, callbacks=[self.early_stop])

    def text_to_sequences(self, text):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters="", oov_token="oov", lower=False)
        self.tokenizer.fit_on_texts([text])
        self.i2w = {index: word for word, index in self.tokenizer.word_index.items()}
        print('Vocabulary Size: %d' % self.vocab_size)
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        sequences = list()
        # create equally-sized windows
        for i in range(self.window, min(self.max_steps, len(encoded) - self.window)):
            sequence = encoded[i - self.window:i + self.window]
            sequences.append(np.array(sequence))
        print('Total Sequences: %d' % len(sequences))
        sequences = np.array(sequences)
        # let the last token from each window be the target
        X, y = sequences[:,:-1], sequences[:,-1]
        # turn y to onehot
        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y

    def generate_next_gram(self, history):
        # encode the text using their UIDs
        encoded = self.tokenizer.texts_to_sequences([history])[0]
        context_encoded = np.array([encoded[- 2 * self.window + 1:]])
        # predict a word from the vocabulary
        if context_encoded.ndim == 1:
            context_encoded = np.array([context_encoded])
        predicted_index = self.model.predict_classes(context_encoded, verbose=0)
        # todo: change the above to the suggested np.argmax(model.predict(x), axis=-1)
        # map predicted word index to word
        next_word = self.i2w[predicted_index[0]]
        return next_word

    # generate a sequence from the model
    def generate_seq(self, seed_text, n_words):
        out_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            out_word = self.generate_next_gram(out_text)
            # append to input
            out_text += " " + out_word
        return out_text

    def compute_gram_probs(self, text):
        """
        The probabilities of the words of the given text.
        :param text: The text the words of which we want to compute the probabilities for.
        :return: A list of probabilities, each in [0,1]
        """
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = 2 * self.window - 1
        probs = []
        for i in range(history, len(encoded)):
            target = encoded[i]
            context_encoded = np.array([encoded[i-history:i]])
            if context_encoded.ndim == 1:
                context_encoded = np.array([context_encoded])
            p = self.model.predict(context_encoded, verbose=0)[0][target]
            probs.append(p)
        return probs

    def predict_words(self, text):
        """
        Mean Reciprocal Rank ( = 1 - Word Error Rate)
        Predict the words of this text, using only the preceding grams.
        :param text: The text to test.
        :return: Return a list of fails/wins (one per word).
        """
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = 2 * self.window - 1
        context_windows = [encoded[i-history:i] for i in range(history, len(encoded))]
        predicted_indices = self.model.predict_classes(context_windows, verbose=0)
        return [None for i in range(history)] + [self.i2w[i] for i in list(predicted_indices)]
        # map predicted word index to word
        #generated_words.append(predicted_index)
        #word_probs = self.model.predict(context_encoded, verbose=0)[0]
        #assert predicted_index == np.argmax(word_probs)
        #p = word_probs[target]
        #next_word = self.i2w[predicted_index]
        #print(f"`{next_word}' was returned; the right word ({self.i2w[target]}) was found at index: {sorted(word_probs).index(p)}")
        #return np.mean(errors)


    def cross_entropy(self, text, PPL=False):
        """
        Cross Entropy of the observed grams. To get the Perplexity (PPL) compute:
        np.power(2, self.cross_entropy(text)).

        :param text: The text to compute BPG for.
        :param PPL: Whether the return the Perplexity score or the cross entropy
        :return: A float number, the lower the better.
        """
        # Get the character probabilities
        probs = self.compute_gram_probs(text)
        # Turn to bits and return bits per character
        log_probs = list(map(np.log2, probs))
        ce = -np.mean(log_probs)
        return np.power(2, ce) if PPL else ce

    def accuracy(self, text, unwanted_term="xxxx", oov="oov", lexicon={}):
        """
        Accuracy of predicting the observed grams.
        :param unwanted_term: if this term is included in a word, ignore.
        :param text: The text to compute the Accuracy.
        :return: A float number; the higher the better.
        """
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        history = 2 * self.window - 1
        scores = []
        keystrokes, keystrokes_discounted = 0,0
        for i in range(history, len(encoded)):
            target = encoded[i]
            target_word = self.i2w[target]
            if unwanted_term in target_word:
                continue
            if target_word == oov or (len(lexicon)>0 and (target_word not in lexicon)):
                scores.append(0)
                keystrokes += len(target_word)
                keystrokes_discounted += len(target_word)
                continue
            context_encoded = encoded[i-history:i]
            #predicted = self.model.predict_classes([context_encoded], verbose=0)[0]
            predicted = np.argmax(self.model.predict([context_encoded], verbose=0), axis=-1)
            if target == predicted:
                scores.append(1)
                keystrokes += len(target_word)
                keystrokes_discounted += 1
            else:
                scores.append(0)
                keystrokes += len(target_word)
                keystrokes_discounted += len(target_word)
        return np.mean(scores), 1-(keystrokes_discounted/keystrokes)
