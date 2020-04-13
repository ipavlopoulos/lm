"""
import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from markov import models as markov_models
from neural import models as neural_models
from scipy.stats import sem
from toolkit import *
from collections import Counter
from absl import flags, logging, app

FLAGS = flags.FLAGS
flags.DEFINE_string("section_name", None, "Empty (default), impression, findings, comparison, indication")
flags.DEFINE_integer("test_size", 100, "Test size.")
flags.DEFINE_integer("vocab_size", 100000, "The size of the vocabulary; rare words are discarded.")
flags.DEFINE_integer("preprocess", 1, "Whether to use pre-processing or not.")
flags.DEFINE_string("dataset_name", "iuxray", "The dataset: iuxray/mimic")
flags.DEFINE_integer("dataset_size", 2928, "The size of the dataset. Default is the small size of iuxray. Assign a "
                                           "large integer to have it in full (e.g., 1000000).")

IUXRAY = "iuxray"
MIMIC = "mimic"


def parse_data(dataset):
    """
    Parse the dataset.
    :param dataset: The dataset name, iuxray or mimic.
    :return: The datadrame with the data.
    """
    assert dataset in {IUXRAY, MIMIC}
    if dataset == IUXRAY:
        data = pd.read_csv(f"./DATA/iuxray.csv")
        data["TEXT"] = data.indication + data.comparison + data.findings + data.impression
        if FLAGS.section_name is not None:
            # Use only a section from each report
            assert FLAGS.section_name in {"indication", "comparison", "findings", "impression"}
            data.TEXT = data[FLAGS.section_name]
    elif dataset == MIMIC:
        data = pd.read_csv("./DATA/NOTEEVENTS.csv.gz")
        # Using only reports about Radiology here
        data = data[data.CATEGORY == "Radiology"]
        if FLAGS.section_name is not None:
            # Use only a section from each report
            assert FLAGS.section_name in {"indication", "comparison", "findings", "impression"}
            sep = f"{FLAGS.section_name.upper()}:"
            data = data[data.TEXT.str.contains(sep)]
            data.TEXT = data.TEXT.apply(lambda report: report.split(sep)[1])
            # todo: this needs improvement, because more than single section will be extracted now
    data = data.dropna(subset=["TEXT"])
    # Keep it fair among datasets, so keep the smallest
    if FLAGS.dataset_size < data.shape[0]:
        print(f"Reducing the data, which originally had {data.shape[0]} texts included.")
        data = data.sample(FLAGS.dataset_size, random_state=42)
    if FLAGS.preprocess == 1:
        data.TEXT = data.TEXT.apply(preprocess)
    data["WORDS"] = data.TEXT.str.split()
    return data


def train_the_ngram_lms(words, kappas=range(1, 9)):
    models =  {}
    for K in kappas:
        ngramlm = markov_models.LM(gram=markov_models.WORD, n=K).train(words)
        models[K] = ngramlm
    return models


def main(argv):
    data = parse_data(FLAGS.dataset_name)
    train, test = train_test_split(data, test_size=FLAGS.test_size, random_state=42)
    #train_words = " ".join(train.TEXT.to_list()).split()
    train_words = train.WORDS.sum()

    # Assess the N-Gram-based LMs
    lms = train_the_ngram_lms(train_words)
    for n in lms:
        print(f"WER({n}-GRAM) @micro:{wer(test.WORDS.sum(), lms[n])}")
        print(f"WER({n}-GRAM) @macro:{test.WORDS.apply(lambda words: wer(words, lms[n])).mean()}")

    # Assess a RNN-based LM
    rnn = neural_models.RNN(epochs=1000)
    rnn.train(train_words)
    print(f"WER(RNNLM) @micro: {1 - rnn.accuracy(' '.join(test.WORDS.sum()))}")
    accuracies = test.WORDS.apply(lambda words: 1 - rnn.accuracy(" ".join(words)))
    print(f'WER(RNNLM) @macro:{accuracies.mean()}±{sem(accuracies.to_list())}')

    # Study the effect of vocabulary size on the best performing 4-Gram-based LM
    test_words = test.WORDS.sum()
    V = Counter(train_words)
    vf_wer = {"V":[], "1-GLM":[], "2-GLM":[], "3-GLM":[], "4-GLM":[], "5-GLM":[], "6-GLM":[], "7-GLM":[], "8-GLM":[],}
    for f in range(10, len(V), 10):
        vf, _ = zip(*V.most_common(f))
        vf_wer["V"].append(f)
        for i in range(1, 9):
            vf_wer[f"{i}-GLM"].append(wer(test_words, lms[i], lexicon=vf))

    vstudy = pd.DataFrame(vf_wer)
    vstudy.to_csv(f"{FLAGS.dataset_name}.vstudy.csv", index=False)

    #ax = vstudy.plot(x="V", title=f"Error Rate in {FLAGS.dataset_name}", ylim=[0.3, 1])
    #ax.set(xlabel="Vocabulary size", ylabel="Error Rate")

    # study what happens only in stop words - use also the RNN

if __name__ == "__main__":
    app.run(main)
