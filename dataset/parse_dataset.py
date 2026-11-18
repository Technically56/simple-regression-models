import argparse
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from base import BaseDataset, get_base_path, perf

if TYPE_CHECKING:
    from ember.features import PEFeatureExtractor

log = logging.getLogger(__name__)

FEATURE_VERSION = 2
VAL_FRACTION = 0.1
_LABEL_RE = re.compile(r'"label":\s*(-?\d+)')


def _labelled_lines(
    paths: Iterable[Path], keep: np.ndarray | None = None
) -> Iterator[str]:
    """Yield labelled lines in file order; with keep (sorted indices), only those."""

    def lines() -> Iterator[str]:
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    m = _LABEL_RE.search(line, 0, 300)
                    if m is None:
                        raise ValueError(f"No label near the start of a line in {p}")
                    if m.group(1) != "-1":
                        yield line

    if keep is None:
        return lines()

    def picked() -> Iterator[str]:
        wanted = iter(keep.tolist())
        target = next(wanted, None)
        for i, line in enumerate(lines()):
            if target is None:
                return
            if i == target:
                yield line
                target = next(wanted, None)

    return picked()


def _sample_rows(
    n_available: int, limit: int | None, rng: np.random.Generator
) -> np.ndarray | None:
    """Sorted random row indices across all files, or None to keep every row.

    The raw files are not shuffled (long runs of one class), so taking the
    first `limit` lines can give a split with a single class.
    """
    if limit is None or limit >= n_available:
        return None
    return np.sort(rng.choice(n_available, size=limit, replace=False))


def _check_both_classes(**splits: np.ndarray) -> None:
    for name, y in splits.items():
        if np.unique(y).size < 2:
            raise ValueError(
                f"The {name} split has only one class (y = {y[0]:.0f}); "
                "use a larger limit."
            )


def _patch_feature_hasher() -> None:
    import ember.features as ef

    if getattr(ef.FeatureHasher, "_str_compat", False):
        return

    class _CompatFeatureHasher(ef.FeatureHasher):
        _str_compat = True

        def transform(self, raw_X):
            return super().transform(
                list(x) if isinstance(x, str) else x for x in raw_X
            )

    ef.FeatureHasher = _CompatFeatureHasher


def _train_stats(x_path: Path, chunk_rows: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path, mmap_mode="r")
    n = len(x)

    shift = np.asarray(x[0], dtype=np.float64)
    s = np.zeros(x.shape[1])
    sq = np.zeros(x.shape[1])
    for i in range(0, n, chunk_rows):
        d = np.asarray(x[i : i + chunk_rows], dtype=np.float64) - shift
        s += d.sum(axis=0)
        sq += (d * d).sum(axis=0)
    del x
    mean_d = s / n
    std = np.sqrt(np.maximum(sq / n - mean_d**2, 0.0))
    std[std == 0] = 1.0
    return (shift + mean_d).astype(np.float32), (1.0 / std).astype(np.float32)


class EmberDataset(BaseDataset):
    name_on_disk = "ember2018"
    _extractor: ClassVar["PEFeatureExtractor | None"] = None

    @staticmethod
    def _init_worker() -> None:
        """Runs once in each Pool worker and gives that worker its own extractor."""
        from ember.features import PEFeatureExtractor

        _patch_feature_hasher()
        EmberDataset._extractor = PEFeatureExtractor(
            feature_version=FEATURE_VERSION, print_feature_warning=False
        )

    @staticmethod
    def _vectorize(line: str) -> tuple[int, np.ndarray]:
        """One JSONL line -> (label, 2381 float32 features). Runs in a worker."""
        extractor = EmberDataset._extractor
        if extractor is None:
            raise RuntimeError(
                "_init_worker did not run in this process; create the Pool with initializer=_init_worker"
            )
        row = json.loads(line)
        return row["label"], extractor.process_raw_features(row)

    @classmethod
    def build(
        cls,
        dataset_path: Path | None = None,
        out_dir: Path | None = None,
        seed: int = 0,
        limit: int | None = None,
        workers: int | None = None,
    ) -> None:

        dataset_path = Path(dataset_path or get_base_path() / "dataset" / "ember2018")
        out_dir = Path(out_dir or cls.processed_dir())
        if cls.is_built(out_dir):
            log.info("%s already built in %s, skipping", cls.__name__, out_dir)
            return
        if out_dir.exists():
            raise FileExistsError(
                f"{out_dir} exists but has no meta.json. Remove it and build again."
            )
        if limit is not None and limit < 10:
            raise ValueError("limit must be at least 10 to create a validation split")
        if workers is not None and workers <= 0:
            raise ValueError("workers must be positive")

        from ember.features import PEFeatureExtractor
        from tqdm import tqdm

        train_files = sorted(dataset_path.glob("train_features_*.jsonl"))
        test_file = dataset_path / "test_features.jsonl"
        if not train_files or not test_file.exists():
            raise FileNotFoundError(
                f"EMBER JSONL files not found in {dataset_path}. Run download_ember() first."
            )

        start = time.perf_counter()
        n_features = PEFeatureExtractor(
            feature_version=FEATURE_VERSION, print_feature_warning=False
        ).dim

        log.info("Counting labelled rows...")
        n_train_avail = sum(1 for _ in _labelled_lines(train_files))
        n_test_avail = sum(1 for _ in _labelled_lines([test_file]))
        sample_rng = np.random.default_rng((seed, 1))
        train_keep = _sample_rows(n_train_avail, limit, sample_rng)
        test_keep = _sample_rows(n_test_avail, limit, sample_rng)
        n_train_all = n_train_avail if train_keep is None else len(train_keep)
        n_test = n_test_avail if test_keep is None else len(test_keep)
        n_val = int(n_train_all * VAL_FRACTION)
        n_train = n_train_all - n_val
        if not n_train or not n_val or not n_test:
            raise ValueError(
                "EMBER must contain nonempty labelled train, validation and test splits"
            )
        log.info("train %d, val %d, test %d rows", n_train, n_val, n_test)

        tmp = out_dir.with_name(out_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)

        def new_x(name: str, rows: int) -> np.memmap:
            return np.lib.format.open_memmap(
                tmp / f"X_{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(rows, n_features),
            )

        x_train, x_val, x_test = (
            new_x("train", n_train),
            new_x("val", n_val),
            new_x("test", n_test),
        )
        y_train = np.empty(n_train, dtype=np.float32)
        y_val = np.empty(n_val, dtype=np.float32)
        y_test = np.empty(n_test, dtype=np.float32)

        perm = np.random.default_rng(seed).permutation(n_train_all)
        processes = workers or max(1, (os.cpu_count() or 2) - 1)
        with Pool(processes=processes, initializer=cls._init_worker) as pool:
            results = pool.imap(
                cls._vectorize, _labelled_lines(train_files, train_keep), chunksize=256
            )
            for k, (label, vec) in enumerate(
                tqdm(results, total=n_train_all, desc="train")
            ):
                slot = perm[k]
                if slot < n_val:
                    x_val[slot], y_val[slot] = vec, label
                else:
                    x_train[slot - n_val], y_train[slot - n_val] = vec, label

            results = pool.imap(
                cls._vectorize, _labelled_lines([test_file], test_keep), chunksize=256
            )
            for k, (label, vec) in enumerate(tqdm(results, total=n_test, desc="test")):
                x_test[k], y_test[k] = vec, label

        for x in (x_train, x_val, x_test):
            x.flush()
        del (
            x,
            x_train,
            x_val,
            x_test,
        )
        _check_both_classes(train=y_train, val=y_val, test=y_test)
        np.save(tmp / "y_train.npy", y_train)
        np.save(tmp / "y_val.npy", y_val)
        np.save(tmp / "y_test.npy", y_test)

        log.info("Computing standardisation stats from the train split...")
        mean, inv_std = _train_stats(tmp / "X_train.npy")
        np.savez(tmp / "stats.npz", mean=mean, inv_std=inv_std)

        meta = {
            "n_features": n_features,
            "n_classes": 2,
            "rows": {"train": n_train, "val": n_val, "test": n_test},
            "seed": seed,
            "feature_version": FEATURE_VERSION,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        tmp.rename(out_dir)

        sec = time.perf_counter() - start
        perf(
            "dataset_build",
            dataset=cls.name_on_disk,
            rows=n_train_all + n_test,
            sec=round(sec, 1),
            workers=processes,
        )
        log.info("Built %s in %s (%.0f s)", cls.name_on_disk, out_dir, sec)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Build memory-mapped EMBER splits (skips existing builds)."
    )
    parser.add_argument(
        "--dataset-path", type=Path, help="Folder containing raw EMBER JSONL files"
    )
    parser.add_argument("--out-dir", type=Path, help="Separate processed output folder")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum labelled rows from each raw train/test source",
    )
    parser.add_argument(
        "--workers", type=int, help="Number of feature extraction processes"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    EmberDataset.build(
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
        limit=args.limit,
        workers=args.workers,
        seed=args.seed,
    )
