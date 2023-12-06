from collections import *
import numpy as np
import random
import pickle
#! pip install nltk==3.5
#nltk.download('punkt')
import nltk

CHARACTER = "CHAR"
WORD = "WORD"


def load(name):
    """
    Load a LM as >>> ngram = models.load("2gram_char_lm.pkl")
    :param name:
    :return:
    """
    with open(name, "rb") as f:
        return pickle.load(f)


def normalize(next_grams_counter):
    """
    For a given n-gram, divide the frequency of each next gram with
    the sum of the frequencies of all the possible next grams.
    :param next_grams_counter: A collections.Counter object about an n-gram
    :return: A list of (gram, probability) tuples.
    """
    s = float(sum(next_grams_counter.values()))
    return [(c, cnt / s) for c, cnt in next_grams_counter.items()]


class LM:
    """
    A language model using n-grams.
    """

    def __init__(self, n=4, gram=CHARACTER):
        assert gram in {CHARACTER, WORD}
        self.gram = gram
        self.n = n
        self.model = {}
        self.name = str(n)+"gram_"+gram.lower()[:4]+"_lm"
        self.vocabulary = {}

    def pad(self):
        if self.gram == CHARACTER:
            return self.n * "~"
        elif self.gram == WORD:
            return self.n * ["~"]
        else:
            raise NotImplemented

    def stringify(self, ngram):
        # If the model is character-based,
        # the ngram is already both a text and a list.
        if self.gram == CHARACTER:
            return ngram
        # If it is word based, join the n words.
        elif self.gram == WORD:
            return " ".join(ngram)

    def train(self, corpus):
        """
        Compute the n-gram statistics on the given corpus.
        :param corpus: The corpus as a single string.
        :return:
        """

        lm = defaultdict(Counter)
        # Initialise also a prior of all words
        self.vocabulary = normalize(Counter(corpus))

        # Let's use a pseudo n-gram to initialise the text
        corpus = self.pad() + corpus
        # Parse the text and store statistics to the dictionary
        for i in range(len(corpus) - self.n):
            prev_grams, next_gram = self.stringify(corpus[i:i + self.n]), corpus[i + self.n]
            lm[prev_grams][next_gram] += 1

        # Update the model with the probabilities of the characters following each possible n-gram.
        self.model = {hist: normalize(next_grams) for hist, next_grams in lm.items()}
        return self

    def compute_gram_probs(self, grams):
        """
        The probabilities of the grams.
        Computed from the statistics observed during training.
        :param grams: The text to compute the probabilities.
        :return: A list of probabilities, each in [0,1]
        """
        # The beginning of the text
        history = self.pad()
        probs = []
        prior = 1.0 / len(self.vocabulary)
        # Parse all text grams
        for i in range(len(grams)):
            # Get the preceding n chars
            past_ngram = self.stringify(history[-self.n:])
            # Get the next gram
            next_gram = grams[i]
            # If the history is unknown, assign zero probability (near zero to make log work)
            if past_ngram not in self.model:
                prob = prior
            else:
                # Get the probability of the next gram
                probable_next_grams = dict(self.model[past_ngram])
                # If it is unknown, assign zero prob
                if next_gram not in probable_next_grams:
                    prob = prior
                else:
                    prob = probable_next_grams[next_gram]
            # Append the probability and update history
            probs.append(prob)
            history = self.stringify(history[-self.n:]) + next_gram
        return probs

    def get_next_grams(self, ngram):
        ngram = self.stringify(ngram)
        if ngram in self.model:
            return self.model[ngram]
        else:
            return self.vocabulary

    def generate_next_gram(self, history, greedy=False, n=1):
        """
        Given an n-gram generate the following gram.
        :param history: A string or word list.
        :param greedy: If true, returns greedily the next most probable next gram.
        :param n: Number of grams to return (1 by default).
        :return: The next gram
        """
        ngram = history[-self.n:]
        # Compute a random number from 0 to 1.
        x = random.random()
        # UNCHECKED
        if greedy:
            if n==1:
                return self.get_next_grams(ngram)[0][0]
            else:
                return self.get_next_grams(ngram)[:n]
        # For each following gram, sort based on freq,
        # and return the most likely gram to follow.

        # If 'random number' - 'gram freq' is below 0
        # return the gram (high-freq gram property).
        if n == 1:
            for c, v in self.get_next_grams(ngram):
                x = x - v
                if x <= 0:
                    return c
        else:
            outputs = []
            for c, v in self.get_next_grams(ngram):
                x = x - v
                if x <= 0:
                    outputs.append((c,v))
                    if len(outputs) == n:
                        return outputs

    def generate_text(self, grams_num=1000):
        """
        Generate a number of grams from scratch.
        :param grams_num: How many grams should the text to be generated have.
        :return: A text of "grams_num" grams.
        """
        # A pseudo text or word-list
        history = self.pad()
        out = []
        # For each time step
        for i in range(grams_num):
            # Use the history to suggest the next gram
            next_gram = self.generate_next_gram(history)
            out.append(next_gram)
            # Update the history with the gram (or [gram] in word-models)
            if self.gram == WORD:
                history = history[-self.n:] + [next_gram]
            else:
                history = history[-self.n:] + next_gram
        sep = " " if self.gram == WORD else ""
        return sep.join(out)

    def bpc(self, text):
        """
        Bits Per Character
        The negative log probabilities of characters (words) averaged across the sequence 
        Perplexity (PPL) can be computed as: np.power(2, self.bpc(text))
        :param text: The text to compute the BPC for
        :return: A float number, the lower the better
        """
        probs = self.compute_gram_probs(text)
        log_probs = list(map(np.log2, probs))
        return -np.mean(log_probs)
    
    def cross_entropy(self, text):
        """ DEPRECATED
        Cross Entropy (aka Bits Per Character or BPC for characters).
        This is the negative mean log prob of the words/characters.
        To get the Perplexity (PPL) compute: np.power(2, self.cross_entropy(text)).
        :param text: The text to compute BPG for.
        :return: A float number, the lower the better.
        """
        # Get the character probabilities
        probs = self.compute_gram_probs(text)
        # Turn to bits and return bits per character
        log_probs = list(map(np.log2, probs))
        return -np.mean(log_probs)

    def save(self):
        with open(self.name+".pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def ppl(self, text):
        """
        Perplexity of a text for the given model.
        That is 2 ^ cross_entropy(text). Bits per gram here is cross entropy.
        :param text:
        :return:
        """
        return np.power(2, self.cross_entropy(text))


class NLTKLM:

    def __init__(self, ngrams_num=4, smoothing="mle"):
        self.ngrams_num = ngrams_num
        self.models = {#"lid": nltk.lm.Lidstone(order=ngrams_num, gamma=0.1),
                       "lap": nltk.lm.Laplace(order=ngrams_num),
                       "mle": nltk.lm.MLE(ngrams_num),
                       "wbi": nltk.lm.WittenBellInterpolated(ngrams_num),
                       "kni": nltk.lm.KneserNeyInterpolated(ngrams_num)
                       }
        assert smoothing in self.models
        self.model = self.models[smoothing]
        self.vocab = None

    def train(self, text):
        train, self.vocab = nltk.lm.preprocessing.padded_everygram_pipeline(self.ngrams_num, [self.preprocess(text)])
        self.model.fit(train, self.vocab)

    def tokenise(self, text):
        return nltk.word_tokenize(text)

    def preprocess(self, tokens):
        return list(nltk.lm.preprocessing.pad_both_ends(tokens, n=self.ngrams_num))

    def ppl(self, text):
        return self.model.perplexity([self.preprocess(text)])
