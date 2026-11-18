from typing import TypedDict

import numpy as np

from base import BaseBackend, BaseDataset, BaseModel
from base.normalise import normalise


class EvaluationMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    n: int


def evaluate(
    model: BaseModel,
    backend: BaseBackend,
    dataset: BaseDataset,
    mean: np.ndarray,
    inv_std: np.ndarray,
    batch_size: int = 4096,
    clip: float | None = None,
) -> EvaluationMetrics:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mean.shape != (dataset.n_features,) or inv_std.shape != mean.shape:
        raise ValueError("Normalisation stats must match the dataset's features")

    mean_device = backend.to_device(mean)
    inv_std_device = backend.to_device(inv_std)
    tn = fp = fn = tp = 0
    for x, y in dataset.batches(batch_size):
        pred = model.predict(normalise(backend, x, mean_device, inv_std_device, clip))
        positive = backend.to_device(y) == 1
        tp += int(backend.to_scalar(backend.sum(pred & positive)))
        fp += int(backend.to_scalar(backend.sum(pred & ~positive)))
        fn += int(backend.to_scalar(backend.sum(~pred & positive)))
        tn += int(backend.to_scalar(backend.sum(~pred & ~positive)))

    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion": [[tn, fp], [fn, tp]],
        "n": n,
    }
