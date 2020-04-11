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
        data = data.sample(FLAGS.data_size, random_state=42)
    if FLAGS.preprocess == 1:
        data.TEXT = data.TEXT.apply(preprocess)
    data["WORDS"] = data.TEXT.str.split()
    return data


def train_the_ngram_lms(words, kappas=range(1, 11)):
    models =  {}
    for K in kappas:
        ngramlm = markov_models.LM(gram=markov_models.WORD, n=K).train(words)
        models[K] = ngramlm
    return models


def main(argv):
    data = parse_data(FLAGS.dataset_name)
    train, test = train_test_split(data, test_size=FLAGS.test_size, random_state=42)

    # get the words
    #words = " ".join(train.TEXT.to_list()).split()
    words = train.WORDS.sum()

    # Get the N-GramLMs
    lms = train_the_ngram_lms(words)
    for n in lms:
        print(f"WER({n}-GRAM) @micro:{wer(test.WORDS.sum(), lms[n])}")
        print(f"WER({n}-GRAM) @macro:{test.WORDS.apply(lambda words: wer(words, lms[n])).mean()}")

    # Get the RNNLM
    rnn = neural_models.RNN(epochs=1000)
    rnn.train(words)
    print(f"WER(RNNLM) @micro: {1 - rnn.accuracy(' '.join(test.WORDS.sum()))}")
    accuracies = test.WORDS.apply(lambda words: 1 - rnn.accuracy(" ".join(words)))
    print(f'WER(RNNLM) @macro:{accuracies.mean()}±{sem(accuracies.to_list())}')


if __name__ == "__main__":
    app.run(main)
