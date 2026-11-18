import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend.math_backend import get_backend
from base import BaseBackend, BaseDataset, BaseModel, get_base_path, perf, setup_logging
from benchmark import EvaluationMetrics, evaluate
from dataset import download_ember
from dataset.parse_dataset import EmberDataset
from training.helpers import MODEL_CLASSES, save_model
from training.train import EpochMetrics, Trainer, TrainerConfig


def create_run_dir(model: str, root: Path | None = None) -> Path:
    """Reserve a separate UTC-labelled folder for each experiment."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    run_dir = (root or get_base_path() / "runs") / f"{stamp}_{model}"
    run_dir.mkdir(parents=True)
    return run_dir


def prepare_data(
    limit_rows: int | None = None, data_dir: Path | None = None
) -> tuple[EmberDataset, EmberDataset, EmberDataset]:
    """Load splits, building a separate quick dataset when a row limit is given.

    An explicit data_dir must already be built. Test rows are always kept in
    full; the row limit applies only to training and validation.
    """
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be positive")
    root = Path(data_dir) if data_dir is not None else EmberDataset.processed_dir()
    if data_dir is None and not EmberDataset.is_built(root):
        if limit_rows is not None:
            root = root.with_name(f"ember2018_quick_{limit_rows}")
        if not EmberDataset.is_built(root):
            raw = get_base_path() / "dataset" / "ember2018"
            if not (raw / "test_features.jsonl").exists():
                download_ember(dest=raw)
            EmberDataset.build(
                out_dir=root, limit=max(10, limit_rows) if limit_rows else None
            )

    train = EmberDataset.load("train", root)
    val = EmberDataset.load("val", root)
    test = EmberDataset.load("test", root)
    if limit_rows is not None:
        train = train.subset(limit_rows)
        val = val.subset(max(1, limit_rows // 5))
    return train, val, test


def run_training(
    cfg: TrainerConfig,
    train: BaseDataset,
    val: BaseDataset,
    test: BaseDataset,
    *,
    run_dir: Path,
    backend: BaseBackend | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> tuple[BaseModel, dict[str, list[float]], EvaluationMetrics]:
    """Shared CLI/UI pipeline: train, restore best weights, save, and test."""
    setup_logging(run_dir)
    backend = get_backend() if backend is None else backend
    perf(
        "run",
        model=cfg.model,
        train_rows=len(train),
        val_rows=len(val),
        test_rows=len(test),
        backend=backend.name,
    )
    model = MODEL_CLASSES[cfg.model](backend, train.n_features, l2=cfg.l2)
    history = Trainer(model, backend, train, val, cfg).train(on_epoch=on_epoch)
    save_model(run_dir / "model.npz", model, cfg, train.mean, train.inv_std)
    metrics = evaluate(
        model, backend, test, train.mean, train.inv_std, cfg.batch_size, cfg.clip
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    perf("test", **metrics)
    return model, history, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate on EMBER 2018.")
    parser.add_argument("--model", choices=MODEL_CLASSES, default="logistic")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Limit standardised features to +/- this value (0 = no clipping)",
    )
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--data-dir", type=Path, help="Existing processed split folder")
    parser.add_argument(
        "--run-dir", type=Path, help="Output folder (default: unique folder in runs/)"
    )
    args = parser.parse_args(argv)
    try:
        cfg = TrainerConfig(
            model=args.model,
            lr=args.lr,
            l2=args.l2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            clip=args.clip or None,
        )
        if args.limit_rows is not None and args.limit_rows <= 0:
            raise ValueError("limit_rows must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = args.run_dir or create_run_dir(cfg.model)
    setup_logging(run_dir)
    train, val, test = prepare_data(args.limit_rows, args.data_dir)
    _, _, metrics = run_training(cfg, train, val, test, run_dir=run_dir)
    print(json.dumps({"run_dir": str(run_dir.resolve()), **metrics}, indent=2))


if __name__ == "__main__":
    main()
