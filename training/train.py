import math
import time
from dataclasses import dataclass

import numpy as np

from base.base_backend import Array, BaseBackend
from base.base_dataset import BaseDataset
from base.base_model import BaseModel
from base.logger import perf


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


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        backend: BaseBackend,
        train_data: BaseDataset,
        val_data: BaseDataset,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.backend = backend
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.mean = backend.to_device(train_data.mean)
        self.inv_std = backend.to_device(train_data.inv_std)

    def _prepare(self, x_in: np.ndarray, y_in: np.ndarray) -> tuple[Array, Array]:
        x = self.backend.to_device(x_in)
        x = self.backend.multiply(self.backend.subtract(x, self.mean), self.inv_std)
        return x, self.backend.to_device(y_in)

    def _validate(self) -> float:
        total_loss = 0.0
        for x_val, y_val in self.val_data.batches(self.cfg.batch_size):
            x_val, y_val = self._prepare(x_val, y_val)
            total_loss += self.backend.to_scalar(self.model.loss(x_val, y_val)) * len(
                y_val
            )
        return total_loss / len(self.val_data)

    def train(self) -> dict[str, list[float]]:
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

            if val_J < best_J:
                best_J, best_state, bad_epochs = val_J, self.model.state(), 0
            else:
                bad_epochs += 1

            history["train_J"].append(train_J)
            history["val_J"].append(val_J)
            perf(
                "epoch",
                epoch=epoch,
                train_J=train_J,
                val_J=val_J,
                best_J=best_J,
                seconds=time.perf_counter() - t0,
            )

            if bad_epochs >= self.cfg.patience:
                perf("early_stop", epoch=epoch, best_J=best_J)
                break

        self.model.load(best_state)
        return history
