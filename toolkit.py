import numpy as np
import re

xxxx = "xxxx"

def fill_unk(lexicon, words_to_fill, pseudo="UNK"):
    """
    Fill rare words with UNK
    :param lexicon: vocabulary
    :param words_to_fill: the words to fill in
    :param pseudo: the pseudo token to substitute unknown words
    :return:
    """
    return [w if w in lexicon else pseudo for w in words_to_fill]


def dedeidentify(text):
    """
    MIMIC-specific
    :param text: The text to de-de-identify
    :return:
    """
    return re.sub(r"\[\*\*.*\*\*\]", xxxx, text).lower()


def preprocess(text):
    """
    Preprocessor of medical texts: mask numbers, and remove some punctuation and symbols.
    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub(r"\d+", "NUMBER", text)
    text = re.sub(r"[()-_*\[\]#\"']+", " ", text)
    text = re.sub(r"[:.,;]+", "", text)
    return text


def wer(words, lm, lexicon=None, ignore_xxxx=True):
    """
    Word Error Rate
    That is the percent of words predicted incorrectly (1-MRR1)
    given the ground truth history

    :param ignore_xxxx: Ignore the de-identification symbol from the errors.
    :param text: the text words
    :param lm: the model
    :param lexicon: score only words from this lexicon (by default empty)
    :return: the score
    """
    results = []
    for i in range(lm.n, len(words)):
        gold_word = words[i]
        if xxxx in gold_word:
            continue
        if lexicon is not None:
            if gold_word not in lexicon:
                continue
        history = words[i-lm.n:i]
        pred_word = lm.generate_next_gram(history)
        results.append(0 if gold_word == pred_word else 1)# count the errors
    return np.mean(results)
