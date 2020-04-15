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
flags.DEFINE_string("section_name", None, "Valid only for IUXRay. Focus to a single section of the report. Examples:"
                                          "'impression', 'findings', 'comparison', 'indication'. Default is None.")
flags.DEFINE_string("report_type", "Radiology", "Valid only for MIMIC-III. Examples: 'Radiology', 'Discharge summary'."
                                                "Default is 'Radiology'.")
flags.DEFINE_integer("test_size", 100, "Test size.")
flags.DEFINE_integer("vocab_size", 1000, "The size of the vocabulary; rare words are discarded.")
flags.DEFINE_integer("preprocess", 1, "Whether to use pre-processing or not.")
flags.DEFINE_string("dataset_name", "iuxray", "The dataset: iuxray/mimic")
flags.DEFINE_integer("dataset_size", 2928, "The size of the dataset. Default is the small size of iuxray. Assign a "
                                           "large integer to have it in full (e.g., 1000000).")
flags.DEFINE_string("method", "neural", "Either neural or ngrams.")
flags.DEFINE_integer("explore_vocab_sensitivity", 0, "Whether to run N-Grams w.r.t. vocabulary size (1) or not (0).")
flags.DEFINE_integer("stopwords_only", 0, "Evaluate only stopwords (1) or pass (0, default).")
flags.DEFINE_integer("repetitions", 5, "Number of repetitions for Monte Carlo Cross Validation.")
flags.DEFINE_string("averaging", "both", "Micro/macro averaging or both (default).")
flags.DEFINE_integer("epochs", 100, "Number of epochs for neural language modeling.")
flags.DEFINE_integer("min_word_freq", 5, "Any words with frequency less than that are masked and ignored.")

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
        data = data[data.CATEGORY == FLAGS.report_type]
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
        print(f"New dataset size: {data.shape[0]}")
    if FLAGS.preprocess == 1:
        data.TEXT = data.TEXT.apply(dedeidentify)
        data.TEXT = data.TEXT.apply(preprocess)
    data["WORDS"] = data.TEXT.str.split()
    return data


def train_the_ngram_lms(words, kappas=range(1, 9)):
    models = {}
    for K in kappas:
        ngramlm = markov_models.LM(gram=markov_models.WORD, n=K).train(words)
        models[K] = ngramlm
    return models


def assess_nglms(datasets, kappas=range(1, 9)):
    acc = {"micro": {k:[] for k in kappas}, "macro":{k:[] for k in kappas}}
    for train_words, test_words, test in datasets:
        # Assess the N-Gram-based LMs
        lms = train_the_ngram_lms(train_words, kappas=kappas)
        for n in lms:
            acc["micro"][n].append(accuracy(test_words, lms[n]))
            if FLAGS.averaging in {"macro", "both"}:
                acc["macro"][n].append(test.WORDS.apply(lambda words: accuracy(words, lms[n])).mean())
    return acc


def assess_lstmlm(datasets):
    print("Setting up the RNNLM...")
    micro, macro = [], []
    for train_words, test_words, test in datasets:
        rnn = neural_models.RNN(epochs=FLAGS.epochs, vocab_size=FLAGS.vocab_size)
        rnn.train(train_words)
        micro.append(rnn.accuracy(' '.join(test_words), unwanted_term=xxxx))
        if FLAGS.averaging not in {"macro", "both"}:
            continue
        macro.append(test.WORDS.apply(lambda words: rnn.accuracy(" ".join(words))).mean())
    return micro, macro


def stopwords_analysis(datasets):
    train_words, test_words, test = datasets[-1]
    # Study what happens when only stop words are used.
    # Use the word list from https://www.textfixer.com/tutorials/common-english-words.txt
    print("Exploring the use on Stop Words...")
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
    print(f"Stop tokens are: {len([w for w in test_words if w in stopwords])} out of {len(test_words)} @test.")
    lms = train_the_ngram_lms(train_words)
    for n in lms:
        micro_ac = accuracy(words=test_words, lm=lms[n], lexicon=stopwords)
        if FLAGS.averaging not in {"macro", "both"}:
            print(f"{n} \t {100 * micro_ac:.2f}")
        macro_ac = test.WORDS.apply(lambda words: accuracy(words=words, lm=lms[n], lexicon=stopwords)).mean()
        print(f"{n} \t {100 * micro_ac:.2f} \t {100 * macro_ac:.2f}")


def vocab_size_sensitivity(datasets, random_choise=-1):
    """
    Train N-Gram LMs (from a random split) and assess different usage.
    That is, only assess words from a lexicon (as if the user was using this only to write stop words).
    :param datasets:
    :param random_choise:
    :return:
    """
    train_words, test_words, test = datasets[random_choise]
    lms = train_the_ngram_lms(train_words)
    print("Investigating the effect of the vocabulary size (using a single split & micro ER)...")
    # Study the effect of vocabulary size on the best performing 4-Gram-based LM
    V = Counter(train_words)
    vf_wer = {"V": [], "1-GLM": [], "2-GLM": [], "3-GLM": [], "4-GLM": [], "5-GLM": [], "6-GLM": [], "7-GLM": [],
              "8-GLM": [], }
    for f in range(50, len(V), 50):
        vf, _ = zip(*V.most_common(f))
        vf_wer["V"].append(f)
        for i in range(1, 9):
            vf_wer[f"{i}-GLM"].append(accuracy(test_words, lms[i], lexicon=vf))

        vstudy = pd.DataFrame(vf_wer)
        vstudy.to_csv(f"{FLAGS.dataset_name}.vstudy.csv", index=False)
        # Plot as follows: >> ax = vstudy.plot(x="V"); ax.set(xlabel="Vocabulary size", ylabel="Error Rate")


def data_explorer(data_pd):
    texts = data_pd.TEXT.sum()
    words = texts.split()
    print(f"Character types #: {len(set(texts))} and occurrences: {len(texts)}")
    print(f"Word types #: {len(set(words))} and occurrences: {len(words)}")
    # todo: add some plot


def main(argv):

    # load the data
    data = parse_data(FLAGS.dataset_name)

    data_explorer(data)

    # perform some exploratory analysis
    # count words and tokens
    # info: perhaps also mask rare words to assist training - to be disregarded during testing

    # create the MC sampled data sets
    datasets = []
    for i in range(FLAGS.repetitions):
        train, test = train_test_split(data, test_size=FLAGS.test_size, random_state=42)
        train_words = train.WORDS.sum()  # " ".join(train.TEXT.to_list()).split()
        test_words = test.WORDS.sum()
        if FLAGS.min_word_freq > 0:
            # mask rare words
            vocab = {w for w, f in Counter(train_words).items() if f > FLAGS.min_word_freq}
            train_words = [w if w in vocab else oov for w in train_words]
            test_words = [w if w in vocab else oov for w in test_words]
        datasets.append((train_words, test_words, test))

    if FLAGS.method == "ngrams":
        # print("Evaluating N-Grams on the test...")
        acc = assess_nglms(datasets)
        for n in range(1, 9):
            print(f"{n}-GLM & " +
                  f"{100 * np.mean(acc['micro'][n]):.2f} ± {100*sem(acc['micro'][n]):.2f} &" +
                  (f"{100 * np.mean(acc['macro'][n]):.2f} ± {100*sem(acc['macro'][n]):.2f} " if FLAGS.averaging in {"macro", "both"} else "") +
                  f"\\\\"
                  )

    if FLAGS.method == "neural":
        micro, macro = assess_lstmlm(datasets)
        print(f"micro:{np.mean(micro)} ± {sem(micro)}, macro:{np.mean(macro)} ± {sem(macro)}")

    if FLAGS.stopwords_only == 1:
        stopwords_analysis(datasets)

    if FLAGS.explore_vocab_sensitivity == 1:
        vocab_size_sensitivity(datasets)


if __name__ == "__main__":
    app.run(main)
