import pandas as pd
from sklearn.model_selection import train_test_split
from markov import models as markov_models
from neural import models as neural_models
from collections import Counter
from scipy.stats import sem
from toolkit import *

# FLAGS
use_impressions_only = True
use_preprocessing = True
train_size = 20000
vocab_size = 100

data = pd.read_csv("./DATA/iuxray.csv")
data["TEXT"] = data.indication + data.comparison + data.findings + data.impression

if use_impressions_only:
    # Use only the IMPRESSION section from each report
    data.TEXT = data.impression

# TRAIN/TEST split
train, test = train_test_split(data, test_size=int(train_size/10), random_state=42)
train = train.dropna(subset=["TEXT"])
test = test.dropna(subset=["TEXT"])

# TRAIN NGRAMs: one on male patients, one on female and one when disregarding gender
# get a sample of the texts per gender
texts = train.TEXT

# preprocess
if use_preprocessing:
    texts = texts.apply(preprocess)


# take the words (efficiently)
words = " ".join(texts.to_list()).split()


# create a vocabulary
WF = Counter(words)
print("|V|:", len(WF))
print("FREQUENT WORDS:", WF.most_common(10))
print("RARE WORDS:", WF.most_common()[:-10:-1])
V, _ = zip(*Counter(words).most_common(vocab_size))
V = set(V)

# substitute any unknown words in the texts
words = fill_unk(words_to_fill=words, lexicon=V)

test["WORDS"] = test.apply(lambda row: fill_unk(V, preprocess(row.TEXT).split()), 1)

# train the N-Grams for N: 1 to 10
for N in range(1, 10):
    wlm = markov_models.LM(gram=markov_models.WORD, n=N).train(words)
    print(f"WER of {N}-GRAM:{test.WORDS.apply(lambda words: wer(words, wlm)).mean()}")

# train RNNLM
rnn = neural_models.RNN(epochs=1000)
rnn.train(words)
print(f"WER of RNNLM:")
accuracies = test.WORDS.apply(lambda words: 1-rnn.accuracy(" ".join(words)))
print(f'@F:{accuracies.mean()}±{sem(accuracies.to_list())}')
