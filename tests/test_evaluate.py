import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cpu_backend import CPUBackend
from benchmark import evaluate
from dataset.parse_dataset import EmberDataset
from training.helpers import MODEL_CLASSES, load_model, save_model
from training.train import TrainerConfig


def make_split(name="test", x=None, y=None):
    if x is None:
        x = np.arange(8, 14, dtype=np.float32).reshape(-1, 1)
    if y is None:
        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
    return EmberDataset(
        name,
        x,
        y,
        np.array([10], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        {
            "n_features": 1,
            "n_classes": 2,
            "rows": {name: len(y)},
            "seed": 0,
            "feature_version": 2,
        },
    )


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.backend = CPUBackend()
        self.data = make_split()

    def test_hand_computed_counts_for_both_models_across_partial_batches(self):
        for name, model_cls in MODEL_CLASSES.items():
            with self.subTest(model=name):
                model = model_cls(self.backend, 1)
                model.load(
                    {
                        "w": np.array([2], dtype=np.float32),
                        "b": np.array(
                            0 if name == "logistic" else 0.5, dtype=np.float32
                        ),
                    }
                )
                result = evaluate(
                    model,
                    self.backend,
                    self.data,
                    self.data.mean,
                    self.data.inv_std,
                    batch_size=4,
                )
                self.assertEqual(result["confusion"], [[1, 2], [1, 2]])
                self.assertEqual(result["n"], 6)
                self.assertEqual(result["accuracy"], 0.5)
                self.assertEqual(result["precision"], 0.5)
                self.assertAlmostEqual(result["recall"], 2 / 3)
                self.assertAlmostEqual(result["f1"], 4 / 7)

    def test_constant_predictions_and_absent_positive_class(self):
        model = MODEL_CLASSES["logistic"](self.backend, 1)
        for bias, confusion, precision, recall in (
            (1, [[0, 3], [0, 3]], 0.5, 1.0),
            (-1, [[3, 0], [3, 0]], 0.0, 0.0),
        ):
            model.load(
                {
                    "w": np.zeros(1, dtype=np.float32),
                    "b": np.array(bias, dtype=np.float32),
                }
            )
            result = evaluate(
                model, self.backend, self.data, self.data.mean, self.data.inv_std, 2
            )
            self.assertEqual(result["confusion"], confusion)
            self.assertEqual(result["precision"], precision)
            self.assertEqual(result["recall"], recall)
        self.data.y[:] = 0
        result = evaluate(
            model, self.backend, self.data, self.data.mean, self.data.inv_std
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_empty_dataset_and_bad_batch_size(self):
        model = MODEL_CLASSES["linear"](self.backend, 1)
        result = evaluate(
            model, self.backend, self.data.subset(0), self.data.mean, self.data.inv_std
        )
        self.assertEqual(
            result,
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion": [[0, 0], [0, 0]],
                "n": 0,
            },
        )
        with self.assertRaises(ValueError):
            evaluate(
                model, self.backend, self.data, self.data.mean, self.data.inv_std, 0
            )

    def test_reload_uses_saved_stats_even_when_dataset_stats_differ(self):
        for name, cls in MODEL_CLASSES.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                model = cls(self.backend, 1)
                model.load(
                    {
                        "w": np.array([2], dtype=np.float32),
                        "b": np.array(0.5, dtype=np.float32),
                    }
                )
                expected = evaluate(
                    model, self.backend, self.data, self.data.mean, self.data.inv_std
                )
                path = Path(tmp) / "model.npz"
                save_model(
                    path,
                    model,
                    TrainerConfig(model=name),
                    self.data.mean,
                    self.data.inv_std,
                )
                loaded, mean, inv_std, cfg = load_model(path, self.backend)
                self.data.mean = np.array([1000], dtype=np.float32)
                self.assertEqual(
                    evaluate(loaded, self.backend, self.data, mean, inv_std), expected
                )
                self.assertEqual(cfg.model, name)
                self.data.mean = np.array([10], dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
