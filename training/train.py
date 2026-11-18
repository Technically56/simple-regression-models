import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from base.base_backend import Array, BaseBackend
from base.base_dataset import BaseDataset
from base.base_model import BaseModel
from base.logger import perf
from base.normalise import normalise


class EpochMetrics(TypedDict):
    epoch: int
    train_J: float
    val_J: float
    best_J: float
    seconds: float
    stopping: bool


@dataclass
class TrainerConfig:
    # Default configuration for training a model.
    model: str = "logistic"
    l2: float = 1e-4
    lr: float = 0.1
    epochs: int = 10
    batch_size: int = 4096
    patience: int = 3
    seed: int = 0
    log_every: int = 50
    clip: float | None = 5.0  # limit standardised features to +/- clip; None = off

    def __post_init__(self):
        if self.model not in {"linear", "logistic"}:
            raise ValueError("model must be linear or logistic")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        for name in ("epochs", "batch_size", "patience", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.clip is not None and (not math.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be positive and finite, or None")


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        backend: BaseBackend,
        train_data: BaseDataset,
        val_data: BaseDataset,
        cfg: TrainerConfig,
    ):
        if not len(train_data) or not len(val_data):
            raise ValueError("Training and validation splits must both have rows")
        self.model = model
        self.backend = backend
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.mean = backend.to_device(train_data.mean)
        self.inv_std = backend.to_device(train_data.inv_std)

    def _prepare(self, x_in: np.ndarray, y_in: np.ndarray) -> tuple[Array, Array]:
        x = normalise(self.backend, x_in, self.mean, self.inv_std, self.cfg.clip)
        return x, self.backend.to_device(y_in)

    def _validate(self) -> float:
        total_loss = 0.0
        for x_val, y_val in self.val_data.batches(self.cfg.batch_size):
            x_val, y_val = self._prepare(x_val, y_val)
            total_loss += self.backend.to_scalar(self.model.loss(x_val, y_val)) * len(
                y_val
            )
        return total_loss / len(self.val_data)

    def train(
        self, on_epoch: Callable[[EpochMetrics], None] | None = None
    ) -> dict[str, list[float]]:
        rng = np.random.default_rng(self.cfg.seed)

        history = {"train_J": [], "val_J": []}

        best_J, best_state, bad_epochs = float("inf"), self.model.state(), 0

        steps_taken = 0

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            running, n_batches = self.backend.zeros(()), 0
            for current_x_batch, current_y_batch in self.train_data.batches(
                self.cfg.batch_size, rng
            ):
                current_x_batch, current_y_batch = self._prepare(
                    current_x_batch, current_y_batch
                )
                J, grads = self.model.gradient(current_x_batch, current_y_batch)
                self.model.step(grads, self.cfg.lr)
                running = self.backend.add(running, J)
                n_batches = n_batches + 1
                steps_taken += 1
                if steps_taken % self.cfg.log_every == 0:
                    perf(
                        "train_step",
                        epoch=epoch,
                        step=steps_taken,
                        J=self.backend.to_scalar(J),
                    )

            train_J = self.backend.to_scalar(running) / n_batches

            if not math.isfinite(train_J):
                raise RuntimeError(f"cost diverged at epoch {epoch}; lower lr")

            val_J = self._validate()
            if not math.isfinite(val_J):
                raise RuntimeError(
                    f"validation cost diverged at epoch {epoch}; lower lr"
                )

            if val_J < best_J:
                best_J, best_state, bad_epochs = val_J, self.model.state(), 0
            else:
                bad_epochs += 1

            history["train_J"].append(train_J)
            history["val_J"].append(val_J)
            record: EpochMetrics = {
                "epoch": epoch,
                "train_J": train_J,
                "val_J": val_J,
                "best_J": best_J,
                "seconds": time.perf_counter() - t0,
                "stopping": bad_epochs >= self.cfg.patience,
            }
            perf("epoch", **record)
            if on_epoch is not None:
                on_epoch(record.copy())

            if bad_epochs >= self.cfg.patience:
                perf("early_stop", epoch=epoch, best_J=best_J)
                break

        self.model.load(best_state)
        return history
