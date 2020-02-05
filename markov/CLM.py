from collections import *
import numpy as np
import random

class clm:
    """
    A character-based language model using n-grams.
    The implementation is partially based on Partially based on nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139.
    """

    def __init__(self, n=4):
        self.n = n
        self.model = {}

    def train(self, text):
        """
        Compute the n-gram statistics on the given corpus.
        :param text: The corpus as a single string.
        :return:
        """
        # Our LM is a number of Counters, one for each available character sequence in our corpus
        lm = defaultdict(Counter)
        # Let's use a pseudo n-gram to initialise the text
        pad = "~" * self.n
        text = pad + text
        # Parse the text and store statistics to the dictionary
        for i in range(len(text) - self.n):
            prev_chars, next_char = text[i:i + self.n], text[i + self.n]
            lm[prev_chars][next_char] += 1

        def normalize(counter):
            """
            Divides the frequency of each next character with
            the sum of the frequencies of all the possible
            next characters for the given n-gram.
            In effect, turns the frequency into a probability.
            :param counter: A collections.Counter object with the
            characters (and their frequencies) following an ngram.
            :return: A list of (char, probability) tuples.
            """
            s = float(sum(counter.values()))
            return [(c, cnt / s) for c, cnt in counter.items()]
        # Update the model with the probabilities of the characters following each possible n-gram.
        self.model = {hist: normalize(chars) for hist, chars in lm.items()}


    def compute_character_probs(self, text):
        """
        The probabilities of the text characters.
        Computed from the character statistics observed during training.
        :param text: The text to compute the probabilities.
        :return: A list of probabilities, each in [0,1]
        """
        # The beginning of the text
        history = "~" * self.n
        probs = []
        # Parse all characters of the text
        for i in range(len(text)):
            # Get the preceding n chars
            past_chars = history[-self.n:]
            # Get the next char
            next_char = text[i]
            # If the history is unknown, assign zero probability (near zero to make log work)
            if past_chars not in self.model:
                prob = 10e-10
            else:
                # Get the probability of the next char in the text
                probable_next_chars = dict(self.model[past_chars])
                # If it is unknown, assign zero prob
                if next_char not in probable_next_chars:
                    prob = 10e-10
                else:
                    prob = probable_next_chars[next_char]
            # Append the probability and update history
            probs.append(prob)
            history = history[-self.n:] + next_char
        return probs

    def bpc(self, text):
        """
        Bits Per Character,
            or the negative mean log prob of the text characters.
        :param text: The text to compute BPC for.
        :return: A float number, the lower the better.
        """
        # Get the character probabilities
        probs = self.compute_character_probs(text)
        # Turn to bits and return bits per character
        log_probs = list(map(np.log2, probs))
        return -np.mean(log_probs)

    def generate_next_letter(self, past_chars):
        """
        Given a sequence of characters generate the character to follow
        :param past_chars: Any string - only the final n chars will be used
        :return: The next char
        """
        past_chars = past_chars[-self.n:]
        # Compute a random number from 0 to 1.
        x = random.random()
        # For each character following the given n-gram,
        # sorted based on frequency (higher first),
        # return the char if the frequency exceeds the random number.
        # Whatever the random number may be, frequent chars
        # are more likely to be emited.
        for c, v in self.model[past_chars]:
            x = x - v
            if x <= 0:
                return c

    def generate_text(self, num_of_chars=1000):
        """
        Generate "num_of_chars" characters starting from the pseudo-token "~"*n
        :param num_of_chars: How many characters should the text to be generated have
        :return: A text of "num_of_chars" characters
        """
        history = "~" * self.n
        out = []
        for i in range(num_of_chars):
            c = self.generate_next_letter(history)
            history = history[-self.n:] + c
            out.append(c)
        return "".join(out)