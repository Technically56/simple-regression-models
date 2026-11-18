import unittest
from unittest.mock import patch

import numpy as np

from backend.cpu_backend import CPUBackend
from models.models import LinearRegressionModel
from test_evaluate import make_split
from training.train import Trainer, TrainerConfig


class CallbackTest(unittest.TestCase):
    def setUp(self):
        self.backend = CPUBackend()
        self.data = make_split("train")
        self.model = LinearRegressionModel(self.backend, 1)

    def test_callback_matches_history_and_is_optional(self):
        cfg = TrainerConfig(model="linear", epochs=3, patience=10, batch_size=4)
        records = []
        history = Trainer(self.model, self.backend, self.data, self.data, cfg).train(
            records.append
        )
        self.assertEqual([r["epoch"] for r in records], [1, 2, 3])
        self.assertEqual([r["train_J"] for r in records], history["train_J"])
        self.assertEqual([r["val_J"] for r in records], history["val_J"])
        self.assertTrue(all(not r["stopping"] and r["seconds"] >= 0 for r in records))
        cfg.epochs = 1
        self.assertEqual(
            len(
                Trainer(self.model, self.backend, self.data, self.data, cfg).train()[
                    "train_J"
                ]
            ),
            1,
        )

    def test_early_stopping_callback_and_best_weights_are_restored(self):
        cfg = TrainerConfig(model="linear", epochs=10, patience=2, batch_size=6)
        trainer = Trainer(self.model, self.backend, self.data, self.data, cfg)
        records, states = [], []

        def capture(record):
            records.append(record)
            states.append(self.model.state())

        with patch.object(trainer, "_validate", side_effect=[0.1, 0.2, 0.3]):
            history = trainer.train(capture)
        self.assertEqual(len(history["train_J"]), 3)
        self.assertEqual([r["stopping"] for r in records], [False, False, True])
        for key, value in self.model.state().items():
            np.testing.assert_array_equal(value, states[0][key])
        self.assertFalse(np.array_equal(states[0]["w"], states[-1]["w"]))

    def test_invalid_config_empty_splits_and_nonfinite_validation(self):
        for kwargs in (
            {"epochs": 0},
            {"batch_size": -1},
            {"patience": 0},
            {"lr": float("nan")},
            {"l2": -1},
            {"model": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainerConfig(**kwargs)
        cfg = TrainerConfig(model="linear", epochs=1)
        with self.assertRaisesRegex(ValueError, "must both have rows"):
            Trainer(self.model, self.backend, self.data.subset(0), self.data, cfg)
        trainer = Trainer(self.model, self.backend, self.data, self.data, cfg)
        with patch.object(trainer, "_validate", return_value=float("nan")):
            with self.assertRaisesRegex(RuntimeError, "validation cost diverged"):
                trainer.train()


if __name__ == "__main__":
    unittest.main()
