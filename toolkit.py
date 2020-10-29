import numpy as np
import re
from tqdm import tqdm

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


def accuracy(words, lm, lexicon={}, relative_kd=True):
    """
    Accuracy in next word prediction. That is the percent of words predicted correctly (a.k.a. MRR1)
    given the ground truth history. Also, return the fraction of keystrokes using assistance to ones not.

    :param relative_kd: if true return keystroke reduction (%), else return (# keystrokes w/, w/o)
    :param lexicon: limited vocabulary to be used during evaluation
    :param words: the text words
    :param lm: the model
    :return: accuracy, keystrokes fraction
    """
    results = []
    strokes, strokes_discounted = 0, 0
    for i in range(lm.n, len(words)):
        gold_word = words[i]
        # fails in OOVs
        if (gold_word == oov) or (len(lexicon)>0 and (gold_word not in lexicon)):
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
    return np.mean(results), 1-(strokes_discounted/strokes) if relative_kd else (strokes_discounted, strokes)


def precision_recall(words, lm, lexicon, neural=False):
    if neural:
        return precision_recall_rnn(words, lm, lexicon)
    else:
        return precision_recall_gram(words, lm, lexicon)


def precision_recall_gram(words, lm, lexicon):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(lm.n, len(words)):
        gold_word = words[i]
        history = words[i - lm.n:i]
        pred_word = lm.generate_next_gram(history)
        if pred_word in lexicon:
            if gold_word == pred_word:
                tp += 1
            else:  # predicted word incorrect but within the lexicon
                fp += 1
        else:
            if gold_word in lexicon:  # gold word in lexicon but not predicted
                fn += 1
            else:  # neither the gold nor the predicted word are in the lexicon
                pass  # or: tn += 1
    return tp/(tp+fp), tp/(tp+fn)


def precision_recall_rnn(rnn, words, lexicon):
    """
    Precision and Recall of the language model, using the lexicon.
    :param rnn: an LSTM or GRU RNN model
    :param words:
    :param lexicon:
    :return:
    """
    encoded = rnn.tokenizer.texts_to_sequences([" ".join(words)])[0]
    history = 2 * rnn.window - 1
    tp, fp, fn, tn = 0, 0, 0, 0
    reference, inputs = [], []
    for i in range(history, len(encoded)):
        target = encoded[i]
        gold_word = rnn.i2w[target]
        reference.append(gold_word)
        context_encoded = encoded[i-history:i]
        inputs.append(context_encoded)
    prediction_scores = rnn.model.predict(inputs, verbose=0)
    pred_indices = np.argmax(prediction_scores, axis=1)
    predictions = [rnn.i2w[pred_index] for pred_index in pred_indices]
    for gold_word, pred_word in tqdm(zip(reference, predictions)):
        if pred_word in lexicon:
            if gold_word == pred_word:
                tp += 1
            else:  # predicted word incorrect but within the lexicon
                fp += 1
        else:
            print (pred_word, gold_word)
            if gold_word in lexicon:  # gold word in lexicon but not predicted
                fn += 1
            else:  # neither the gold nor the predicted word are in the lexicon
                pass  # or: tn += 1
    return tp / (tp + fp + 0.001), tp / (tp + fn + 0.001)
