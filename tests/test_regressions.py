import os
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import toolkit
from markov import models as markov_models


neural_stub = types.ModuleType("neural")
neural_models_stub = types.ModuleType("neural.models")
neural_stub.models = neural_models_stub
sys.modules.setdefault("neural", neural_stub)
sys.modules.setdefault("neural.models", neural_models_stub)

matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
for name in ["figure", "plot", "ylim", "xlabel", "legend", "savefig"]:
    setattr(pyplot_stub, name, mock.Mock())
matplotlib_stub.pyplot = pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

import run_mlm


class PrecisionRecallTests(unittest.TestCase):
    def test_neural_dispatch_preserves_argument_order(self):
        words = ["alpha", "beta"]
        lm = object()
        lexicon = {"beta"}

        with mock.patch("toolkit.precision_recall_rnn", return_value=(0.5, 0.25)) as patched:
            result = toolkit.precision_recall(words=words, lm=lm, lexicon=lexicon, neural=True)

        self.assertEqual(result, (0.5, 0.25))
        patched.assert_called_once_with(lm, words, lexicon)

    def test_precision_recall_gram_handles_zero_denominators(self):
        class FakeLM:
            n = 1

            def generate_next_gram(self, history):
                return "predicted"

        precision, recall = toolkit.precision_recall_gram(
            words=["context", "gold"],
            lm=FakeLM(),
            lexicon={"medical"},
        )

        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)


class RunMlmTests(unittest.TestCase):
    def test_pr_tracks_medical_recall_and_handles_small_lexicons(self):
        train_words = ["stopterm", "medicalterm", "other"]
        test_words = ["a", "b", "c"]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "pr.medsVstop.png")
            lm = mock.Mock()
            lm.name = "3gram_word_lm"
            with mock.patch("run_mlm.precision_recall", side_effect=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]):
                with mock.patch("matplotlib.pyplot.savefig") as savefig:
                    results = run_mlm.pr(
                        train_words=train_words,
                        test_words=test_words,
                        lm=lm,
                        stopwords={"stopterm"},
                        medwords={"medicalterm"},
                        steps=[1, 2],
                    )

        self.assertEqual(list(results["v"]), [1, 2])
        self.assertIn("R@medical", results.columns)
        self.assertEqual(list(results["R@medical"]), [0.4, 0.8])
        savefig.assert_called_once()


class MarkovModelTests(unittest.TestCase):
    def test_word_compute_gram_probs_keeps_token_history(self):
        lm = markov_models.LM(n=2, gram=markov_models.WORD).train(["a", "b", "c", "d"])

        probs = lm.compute_gram_probs(["a", "b", "c", "d"])

        self.assertEqual(probs, [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
