import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def write_splits(root):
    root.mkdir(parents=True)
    x = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = (x[:, 0] >= 12).astype(np.float32)
    for name in ("train", "val", "test"):
        np.save(root / f"X_{name}.npy", x)
        np.save(root / f"y_{name}.npy", y)
    np.savez(root / "stats.npz", mean=x.mean(axis=0), inv_std=1 / x.std(axis=0))
    (root / "meta.json").write_text(
        json.dumps(
            {
                "n_features": 1,
                "n_classes": 2,
                "rows": {name: 24 for name in ("train", "val", "test")},
                "seed": 0,
                "feature_version": 2,
            }
        )
    )


class PipelineTest(unittest.TestCase):
    def test_cli_trains_saves_evaluates_and_logs_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_splits(root / "data")
            for name in ("linear", "logistic"):
                with self.subTest(model=name):
                    run = root / name
                    result = subprocess.run(
                        [
                            sys.executable,
                            "main.py",
                            "--model",
                            name,
                            "--data-dir",
                            str(root / "data"),
                            "--run-dir",
                            str(run),
                            "--epochs",
                            "3",
                            "--batch-size",
                            "5",
                            "--limit-rows",
                            "20",
                        ],
                        cwd=REPO_ROOT,
                        env={**os.environ, "MATH_BACKEND": "numpy"},
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    metrics = json.loads(result.stdout)
                    self.assertEqual(metrics["n"], 24)  # row limit never truncates test
                    for filename in (
                        "model.npz",
                        "run.log",
                        "metrics.jsonl",
                        "history.json",
                        "test_metrics.json",
                    ):
                        self.assertTrue((run / filename).is_file(), filename)
                    events = [
                        json.loads(line)
                        for line in (run / "metrics.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(events[-1]["event"], "test")
                    self.assertEqual(events[-1]["n"], 24)
                    self.assertEqual(
                        len([e for e in events if e["event"] == "epoch"]), 3
                    )


if __name__ == "__main__":
    unittest.main()
