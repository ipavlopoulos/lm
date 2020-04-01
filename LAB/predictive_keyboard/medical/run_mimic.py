import pandas as pd
from sklearn.model_selection import train_test_split
from markov import models as markov_models
from neural import models as neural_models
from collections import Counter
from scipy.stats import sem
from toolkit import *

# FLAGS
use_radiology_only = True
use_impressions_only = True
use_gender_split = True
use_preprocessing = True
train_size = 10000
vocab_size = 1000
N = 3


# PARSE MIMIC
notes = pd.read_csv("./DATA/NOTEEVENTS.csv.gz")
patients = pd.read_csv("./DATA/PATIENTS.csv.gz")
admissions = pd.read_csv("./DATA/ADMISSIONS.csv.gz")
diagnoses = pd.read_csv("./DATA/DIAGNOSES_ICD.csv.gz")
icds = diagnoses.groupby(by=["subject_id".upper()]).ICD9_CODE.apply(list)
# add gender information
data = pd.merge(left=notes[["SUBJECT_ID", "CATEGORY", "TEXT"]], right=patients[["SUBJECT_ID", "GENDER"]], left_on="SUBJECT_ID", right_on="SUBJECT_ID")
# add ICD9 codes
# DATA = pd.merge(left=DATA[["SUBJECT_ID", "CATEGORY", "TEXT", "GENDER"]],
#                right=icds, left_on="SUBJECT_ID", right_on="SUBJECT_ID")

# print some gender-based stats
print(data.GENDER.value_counts())

# filter the DATA
if use_radiology_only:
    # Use only reports about Radiology
    data = data[data.CATEGORY == "Radiology"]

if use_impressions_only:
    # Use only the IMPRESSION section from each report
    data = data[data.TEXT.str.contains("IMPRESSION:")]
    data.TEXT = data.TEXT.apply(lambda report: report.split("IMPRESSION:")[1])

# TRAIN/TEST split
if use_gender_split:
    data_m = data[data.GENDER == "M"]
    data_f = data[data.GENDER == "F"]
    data = pd.concat([data_m.sample(train_size*5), data_f.sample(train_size*5)]).sample(frac=1).reset_index(drop=True)

train = data.sample(train_size*2)
train, test = train_test_split(train, test_size=int(train_size/10), random_state=42)


# TRAIN NGRAMs: one on male patients, one on female and one when disregarding gender
# get a sample of the texts per gender
texts = train.sample(int(train_size/2)).TEXT
texts_m = train[train.GENDER == "M"].TEXT
texts_f = train[train.GENDER == "F"].TEXT

# preprocess
if use_preprocessing:
    texts = texts.apply(preprocess)
    texts_m = texts_m.apply(preprocess)
    texts_f = texts_f.apply(preprocess)


# take the words (efficiently)
words = " ".join(texts.to_list()).split()
words_m = " ".join(texts_m.to_list()).split()
words_f = " ".join(texts_f.to_list()).split()

# create a vocabulary
WF = Counter(words)
print("|V|:", len(WF))
print("FREQUENT WORDS:", WF.most_common(10))
print("RARE WORDS:", WF.most_common()[:-10:-1])
V, _ = zip(*Counter(words).most_common(vocab_size))
V = set(V)

# substitute any unknown words in the texts
words = fill_unk(words_to_fill=words, lexicon=V)
words_m = fill_unk(words_to_fill=words_m, lexicon=V)
words_f = fill_unk(words_to_fill=words_f, lexicon=V)

test["WORDS"] = test.apply(lambda row: fill_unk(V, preprocess(row.TEXT).split()), 1)

# train the n-grams
for name, dataset in (("gender-agnostic", words), ("male", words_m), ("female", words_f)):
    wlm = markov_models.LM(gram=markov_models.WORD, n=N).train(dataset)
    print(f"WER of {name}-NGRAM:")
    print(f'@M:{test[test.GENDER == "M"].WORDS.apply(lambda words: wer(words, wlm))}')
    print(f'@F:{test[test.GENDER == "F"].WORDS.apply(lambda words: wer(words, wlm))}')
# >>> WER of MALE-NGRAM: (0.8298521221038996, 0.8174244389937317)
# >>> WER of FEMALE-NGRAM: (0.8252274537647764, 0.8133000318371052)
# >>> WER of ANY-NGRAM: (0.8373151970659237, 0.8264538956122482)

# Run a Neural LM
rnn = neural_models.RNN(epochs=10)
rnn.train(words)
print(f"WER of RNNLM:")
male_accuracies = test[test.GENDER == "M"].WORDS.apply(lambda words: 1-rnn.accuracy(" ".join(words)))
female_accuracies = test[test.GENDER == "F"].WORDS.apply(lambda words: 1-rnn.accuracy(" ".join(words)))
print(f'@M:{male_accuracies.mean()}±{sem(female_accuracies)}')
print(f'@F:{female_accuracies.mean()}±{sem(female_accuracies)}')
# >>> @M:0.636
# >>> @F:0.626