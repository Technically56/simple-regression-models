import dataclasses
import json
from pathlib import Path

import numpy as np

from base import BaseBackend, BaseModel
from models.models import LinearRegressionModel, LogisticRegressionModel
from training.train import TrainerConfig

MODEL_CLASSES: dict[str, type[BaseModel]] = {
    "linear": LinearRegressionModel,
    "logistic": LogisticRegressionModel,
}


def save_model(
    path: str | Path,
    model: BaseModel,
    cfg: TrainerConfig,
    mean: np.ndarray,
    inv_std: np.ndarray,
) -> None:
    state = model.state()
    np.savez(
        path,
        w=state["w"],
        b=state["b"],
        mean=mean,
        inv_std=inv_std,
        config=json.dumps(dataclasses.asdict(cfg)),
    )


def load_model(
    path: str | Path, backend: BaseBackend
) -> tuple[BaseModel, np.ndarray, np.ndarray, TrainerConfig]:
    """Rebuild a saved model on the given backend.

    Returns (model, mean, inv_std, cfg). Inputs must be normalised with mean/inv_std
    before they are passed to model.predict.
    """
    with np.load(path) as f:
        raw_cfg = json.loads(str(f["config"]))
        # Runs saved before clipping existed were trained without it.
        raw_cfg.setdefault("clip", None)
        cfg = TrainerConfig(**raw_cfg)
        model = MODEL_CLASSES[cfg.model](backend, f["w"].shape[0], l2=cfg.l2)
        model.load({"w": f["w"], "b": f["b"]})
        return model, f["mean"], f["inv_std"], cfg
