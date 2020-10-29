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
from ast import literal_eval
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string("section_name", None, "Valid only for IUXRay. Focus to a single section of the report. Examples:"
                                          "'impression', 'findings', 'comparison', 'indication'. Default is None.")
flags.DEFINE_string("report_type", "Radiology", "Valid only for MIMIC-III. Examples: 'Radiology', 'Discharge summary'."
                                                "Default is 'Radiology'.")
flags.DEFINE_integer("test_size", 10000, "Number of words to test.")
flags.DEFINE_integer("vocab_size", 1000, "The size of the vocabulary; rare words are discarded.")
flags.DEFINE_integer("preprocess", 1, "Whether to use pre-processing or not.")
flags.DEFINE_string("dataset_name", "iuxray", "The dataset: iuxray/mimic")
flags.DEFINE_integer("dataset_size", 2928, "The size of the dataset. Default is the small size of iuxray. Assign a "
                                           "large integer to have it in full (e.g., 1000000).")
flags.DEFINE_string("method", "explore", "One of lstm/gru/counts/explore.")
flags.DEFINE_integer("explore_vocab_sensitivity", 0, "Whether to asses 4GLM (1) GRU (2) or none (0).")
flags.DEFINE_integer("stopwords_only", 0, "Evaluate only stopwords (1) or pass (0, default).")
flags.DEFINE_integer("repetitions", 5, "Number of repetitions for Monte Carlo Cross Validation.")
flags.DEFINE_integer("epochs", 100, "Number of epochs for neural language modeling.")
flags.DEFINE_integer("min_word_freq", 10, "Any words with frequency less than that are masked and ignored.")
flags.DEFINE_integer("max_chars", 10000, "Use only texts with less characters than this number.")
flags.DEFINE_integer("step", 1, "Valid only when a lexicon is given or inferred (explore_vocab_sensitivity>0).")
flags.DEFINE_integer("save_datasets", 0, "Whether to save the datasets (1), sampled but not pre-processed.")
flags.DEFINE_integer("load_datasets", 0, "Whether to load saved  datasets.")
flags.DEFINE_string("stop_lexicon_path", "", "The path to a CSV lexicon of stopwords, with a column 'term' incl. the terms.")
flags.DEFINE_string("meds_lexicon_path", "", "The path to a CSV lexicon of health-related terms, with a column 'term' incl. the terms.")

IUXRAY = "iuxray"
MIMIC = "mimic"


def load_data(name):
    assert FLAGS.load_datasets
    data = pd.read_csv(f"{name}.{FLAGS.report_type[:5].lower()}.csv.gz")
    data.WORDS = data.WORDS.apply(literal_eval)
    return data


def parse_data(dataset):
    """
    Parse the dataset.
    :param dataset: The dataset name, iuxray or mimic.
    :return: The datadrame with the DATA.
    """
    assert dataset in {IUXRAY, MIMIC}
    if dataset == IUXRAY:
        data = pd.read_csv(f"./DATA/iuxray.csv.gz")
        data["TEXT"] = data.indication + data.comparison + data.findings + data.impression
        if FLAGS.section_name is not None:
            # Use only a section from each report
            assert FLAGS.section_name in {"indication", "comparison", "findings", "impression"}
            data.TEXT = data[FLAGS.section_name]
    elif dataset == MIMIC:
        data = pd.read_csv("./DATA/NOTEEVENTS.csv.gz")
        data = data[data.CATEGORY == FLAGS.report_type]
        data = data[data.TEXT.apply(len) < FLAGS.max_chars]
        if FLAGS.section_name is not None:
            # Use only a section from each report
            assert FLAGS.section_name in {"indication", "comparison", "findings", "impression"}
            sep = f"{FLAGS.section_name.upper()}:"
            data = data[data.TEXT.str.contains(sep)]
            data.TEXT = data.TEXT.apply(lambda report: report.split(sep)[1])
            # todo: this needs improvement, because more than single section will be extracted now
    elif "/" in dataset: # path provided to the dataset in CSV format
        data = pd.read_csv(dataset)
    data = data.dropna(subset=["TEXT"])
    # Keep it fair among datasets, so keep the smallest
    if FLAGS.dataset_size < data.shape[0]:
        print(f"Reducing the DATA, which originally had {data.shape[0]} texts included.")
        data = data.sample(FLAGS.dataset_size, random_state=42)
        print(f"New dataset size: {data.shape[0]}")
    if FLAGS.preprocess == 1:
        data.TEXT = data.TEXT.apply(dedeidentify)
        data.TEXT = data.TEXT.apply(preprocess)
    data["WORDS"] = data.TEXT.str.split()
    if FLAGS.save_datasets == 1:
        print("Saving the DATA...")
        data.to_csv(f"{dataset}.{FLAGS.report_type[:5].lower()}.csv", index=False)
    return data


def vocab_size_sensitivity(train_words, test_words, lm, step=150, lexicon=None):
    if lexicon is None:
        vocabulary = Counter(train_words)
    else:
        vocabulary = Counter([w for w in train_words if w in lexicon])

    results = {"V": [], "Accuracy": [], "KD": [], "K": []}
    for f in range(step, FLAGS.vocab_size, step):
        results["V"].append(f)
        lexicon, _ = zip(*vocabulary.most_common(f))
        acc, (kd, keystrokes) = accuracy(words=test_words, lm=lm, lexicon=set(lexicon), relative_kd=False) \
            if "gram" in lm.name else lm.accuracy(" ".join(test_words), lexicon=set(lexicon), relative_kd=False)
        results[f"Accuracy"].append(acc)
        results[f"KD"].append(kd)
        results[f"K"].append(keystrokes)
    results_pd = pd.DataFrame(results)
    results_pd.to_csv(f"{FLAGS.dataset_name}.{lm.name}.vocabulary_study.csv", index=False)
    return results_pd


def pr(train_words, test_words, lm, stopwords, medwords, steps=list(range(2, 10, 2))):
    # adapt the lexicon to the dataset
    v_stop = Counter([w for w in train_words if w in stopwords])
    v_meds = Counter([w for w in train_words if w in medwords])
    meds = "medical"
    stop = "stop"
    results = {"v": [], f"P@{stop}": [], f"R@{stop}": [], f"P@{meds}":[], f"R@{stop}":[]}
    for step in steps:
        results["v"].append(step)
        # get the most commonly used words of the lexicon
        lex_stop, _ = zip(*v_stop.most_common(step))
        lex_meds, _ = zip(*v_meds.most_common(step))
        p_stop, r_stop = precision_recall(words=test_words, lm=lm, lexicon=set(lex_stop), neural="gram" not in lm.name)
        results[f"P@{stop}"].append(p_stop)
        results[f"R@{stop}"].append(r_stop)
        p_meds, r_meds = precision_recall(words=test_words, lm=lm, lexicon=set(lex_meds), neural="gram" not in lm.name)
        results[f"P@{meds}"].append(p_meds)
        results[f"R@{meds}"].append(r_meds)
    results_pd = pd.DataFrame(results)

    _ = plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(results_pd.v, results_pd[f"P@{stop}"], 'b:', label=f"P@{stop}")
    plt.plot(results_pd.v, results_pd[f"P@{meds}"], 'b', label=f"P@{meds}")
    #plt.fill_between(results_pd.v, results_pd["P@stop"], results_pd["P@meds"], color='b', alpha='0.1')
    plt.plot(results_pd.v, results_pd[f"R@{stop}"], 'r:', label=f"R@{stop}")
    plt.plot(results_pd.v, results_pd[f"R@{meds}"], 'r', label=f"R@{meds}")
    #plt.fill_between(results_pd.v, results_pd["R@stop"], results_pd["R@meds"], color='r', alpha='0.1')
    plt.ylim([0, 1])
    plt.xlabel("# top terms assessed")
    plt.legend()
    plt.savefig("pr.medsVstop.png")
    return results_pd


def main(argv):

    # load the DATA
    if FLAGS.load_datasets:
        data = load_data(FLAGS.dataset_name)
    else:
        data = parse_data(FLAGS.dataset_name)

    print("Data loaded...")

    # exploratory analysis
    dist = data.WORDS.apply(len)
    print(f"# words: {dist.mean()} ± {dist.std()} and max: {dist.max()}")
    words = Counter(data.WORDS.sum())
    print(words.most_common(10))

    if FLAGS.method == "explore":
        if FLAGS.explore_vocab_sensitivity == 0:
            return

    # perform some exploratory analysis
    # count words and tokens
    # info: perhaps also mask rare words to assist training - to be disregarded during testing

    # create the MC sampled DATA sets
    datasets = []
    for i in range(FLAGS.repetitions):
        train_test = data.sample(FLAGS.dataset_size, random_state=42+i).WORDS.sum()
        train_words = train_test[:-FLAGS.test_size]
        test_words = train_test[-FLAGS.test_size:] # last words
        if FLAGS.min_word_freq > 0:
            # mask rare words
            vocab = {w for w, f in Counter(train_words).items() if f > FLAGS.min_word_freq}
            train_words = [w if w in vocab else oov for w in train_words]
            test_words = [w if w in vocab else oov for w in test_words]
        datasets.append((train_words, test_words))

    if FLAGS.method == "counts":
        print(f"Evaluating N-Grams on {FLAGS.test_size} unseen words...")
        kappas = range(1, 9)
        accs = {k: [] for k in kappas}
        keys = {k: [] for k in kappas}
        for train_words, test_words in datasets:
            lms = {K:markov_models.LM(gram=markov_models.WORD, n=K).train(train_words) for K in kappas}
            for n in lms:
                acc, kf = accuracy(test_words, lms[n])
                accs[n].append(acc)
                keys[n].append(kf)

        for K in kappas:
            print(f"Micro Accuracy({K}-GLM) & {100 * np.mean(accs[K]):.2f} ± {100*sem(accs[K]):.2f} \\\\")
            print(f"Keystrokes({K}-GLM) & {100 * np.mean(keys[K]):.2f} ± {100*sem(keys[K]):.2f} \\\\")
            print()

    if FLAGS.method in {"lstm", "gru"}:
        print("Setting up the RNNLM...")
        accs, keystrokes = [], []
        for train_words, test_words in datasets:
            rnn = neural_models.RNN(epochs=FLAGS.epochs, vocab_size=FLAGS.vocab_size,
                                    use_gru=int(FLAGS.method == "gru"))
            rnn.train(train_words)
            acc, kf = rnn.accuracy(' '.join(test_words), unwanted_term=xxxx)
            accs.append(acc)
            keystrokes.append(kf)
        print(f"Micro Accuracy:{np.mean(accs)} ± {sem(accs)}")
        print(f"Keystrokes:{np.mean(keystrokes)} ± {sem(keystrokes)}")

    if FLAGS.explore_vocab_sensitivity != 0:
        train_words, test_words = datasets[-1]
        if FLAGS.explore_vocab_sensitivity == 1:
            lm = markov_models.LM(gram=markov_models.WORD, n=3).train(train_words)
        elif FLAGS.explore_vocab_sensitivity == 2:
            lm = neural_models.RNN(epochs=FLAGS.epochs, vocab_size=FLAGS.vocab_size, use_gru=int(FLAGS.method == "gru"))
            lm.train(train_words)
            lm.save()

        if len(FLAGS.stop_lexicon_path) > 0 and len(FLAGS.meds_lexicon_path) > 0:
            print("Exploring P/R of the given lexicons...")
            stop_lexicon = set(pd.read_csv(FLAGS.stop_lexicon_path).term.to_list())
            meds_lexicon = set(pd.read_csv(FLAGS.meds_lexicon_path).term.to_list())

            results_pd = pr(train_words, test_words, lm=lm, stopwords=stop_lexicon, medwords=meds_lexicon)
        else:
            print("Exploring different Vocabulary sizes...")
            _ = vocab_size_sensitivity(train_words, test_words, lm)


if __name__ == "__main__":
    app.run(main)
