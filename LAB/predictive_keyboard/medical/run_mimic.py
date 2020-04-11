import pandas as pd
from sklearn.model_selection import train_test_split
from markov import models as markov_models
from neural import models as neural_models
from collections import Counter
from scipy.stats import sem
from toolkit import *

# todo: add FLAGS
if __name__ == "main":
    use_radiology_only = True
    use_impressions_only = False
    use_preprocessing = True
    test_size = 100
    train_size = 2853
    vocab_size = 10000


    # PARSE MIMIC
    data = pd.read_csv("./DATA/NOTEEVENTS.csv.gz")

    # filter the DATA
    if use_radiology_only:
        # Use only reports about Radiology
        data = data[data.CATEGORY == "Radiology"]

    if use_impressions_only:
        # Use only the IMPRESSION section from each report
        data = data[data.TEXT.str.contains("IMPRESSION:")]
        data.TEXT = data.TEXT.apply(lambda report: report.split("IMPRESSION:")[1])

    data = data.sample(train_size+test_size, random_state=42)
    train, test = train_test_split(data, test_size=test_size, random_state=42)

    texts = train.TEXT

    # preprocess
    if use_preprocessing:
        texts = texts.apply(preprocess)

    # take the words (efficiently)
    words = " ".join(texts.to_list()).split()

    # create a vocabulary
    WF = Counter(words)
    print("|V|:", len(WF), "(before reducing the vocabulary)")
    print("FREQUENT WORDS:", WF.most_common(10))
    print("RARE WORDS:", WF.most_common()[:-10:-1])
    V, _ = zip(*Counter(words).most_common(vocab_size))
    V = set(V)

    # substitute any unknown words in the texts
    _words = fill_unk(words_to_fill=words, lexicon=V)
    assert len(set(words)) == len(set(_words))
    test["WORDS"] = test.apply(lambda row: fill_unk(lexicon=V, words_to_fill=preprocess(row.TEXT).split()), 1)

    # train the N-Grams for N: 1 to 10
    for N in range(1, 10):
        wlm = markov_models.LM(gram=markov_models.WORD, n=N).train(words)
        print(f"WER of {N}-GRAM:{test.WORDS.apply(lambda words: wer(words, wlm)).mean()}")

    # Run a Neural LM
    rnn = neural_models.RNN(epochs=1000)
    rnn.train(words)
    accuracies = test.WORDS.apply(lambda words: 1 - rnn.accuracy(" ".join(words)))
    print(f'WER(RNNLM):{accuracies.mean()}±{sem(accuracies.to_list())}')
