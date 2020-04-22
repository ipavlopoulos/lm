import numpy as np
import re

xxxx = "xxxx"
oov = "oov"

# Use the word list from https://www.textfixer.com/tutorials/common-english-words.txt
stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
             "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
             "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
             "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
             "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
             "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
             "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
             "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
             "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
             "will", "just", "don", "should", "now"}


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


def accuracy(words, lm):
    """
    Accuracy in next word prediction. That is the percent of words predicted correctly (a.k.a. MRR1)
    given the ground truth history. Also, return the fraction of keystrokes using assistance to ones not.

    :param words: the text words
    :param lm: the model
    :return: accuracy, keystrokes fraction
    """
    results = []
    strokes, strokes_discounted = 0, 0
    for i in range(lm.n, len(words)):
        gold_word = words[i]
        # fails in OOVs
        if gold_word == oov:
            results.append(0)
            strokes_discounted += len(gold_word)
            strokes += len(gold_word)
            continue
        # ignore XXXXs
        if xxxx in gold_word:
            continue
        history = words[i-lm.n:i]
        pred_word = lm.generate_next_gram(history)
        # save strokes when succeed
        if gold_word == pred_word:
            results.append(1)
            strokes_discounted += 1.
            strokes += len(gold_word)
        else:
            results.append(0)
            strokes_discounted += len(gold_word)
            strokes += len(gold_word)
    return np.mean(results), 1-(strokes_discounted/strokes)
