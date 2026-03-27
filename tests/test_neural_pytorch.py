import os
import tempfile
import unittest


class NeuralPyTorchStaticTests(unittest.TestCase):
    def test_neural_module_references_pytorch_not_tensorflow(self):
        with open("neural/models.py", "r", encoding="utf8") as handle:
            source = handle.read()

        self.assertIn("import torch", source)
        self.assertIn("class TorchRNNModel", source)
        self.assertNotIn("tensorflow.keras", source)
        self.assertIn('torch.save(checkpoint, f"{name}.pt")', source)


@unittest.skipUnless(os.environ.get("RUN_TORCH_TESTS") == "1", "PyTorch runtime tests are opt-in in constrained environments.")
class NeuralPyTorchIntegrationTests(unittest.TestCase):
    def test_rnn_train_predict_and_save_load(self):
        from neural import models as neural_models

        text = "alpha beta gamma alpha beta gamma alpha beta gamma"
        rnn = neural_models.RNN(
            epochs=1,
            batch_size=2,
            vocab_size=32,
            hidden_size=8,
            window=2,
            split=0.0,
        )

        rnn.train(text)
        prediction = rnn.generate_next_gram("alpha beta gamma")
        probs = rnn.compute_gram_probs("alpha beta gamma alpha beta")

        self.assertIsInstance(prediction, str)
        self.assertTrue(len(probs) > 0)
        self.assertTrue(all(0.0 <= prob <= 1.0 for prob in probs))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "tiny_rnn")
            rnn.save(model_path)
            restored = neural_models.load(model_path)

        restored_prediction = restored.generate_next_gram("alpha beta gamma")
        self.assertIsInstance(restored_prediction, str)


if __name__ == "__main__":
    unittest.main()
