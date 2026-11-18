import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Literal, Self, TypedDict

import numpy as np
from numpy.typing import NDArray

from .base_path import get_base_path


type SplitName = Literal["train", "val", "test"]


class DatasetMeta(TypedDict):
    n_features: int
    n_classes: int
    rows: dict[str, int]
    seed: int
    feature_version: int


class BaseDataset(ABC):
    name_on_disk: ClassVar[str]

    def __init__(
        self,
        name: str,
        x: NDArray[np.float32],
        y: NDArray[np.float32],
        mean: NDArray[np.float32],
        inv_std: NDArray[np.float32],
        meta: DatasetMeta,
    ):
        self.name = name
        self.x = x
        self.y = y
        self.mean = mean
        self.inv_std = inv_std
        self.meta = meta

    @classmethod
    def load(cls, name: SplitName, root: Path | None = None) -> Self:
        """Open one split from the processed folder. X stays on disk as a memmap."""
        root = cls.processed_dir() if root is None else Path(root)
        if not cls.is_built(root):
            raise FileNotFoundError(
                f"No processed {cls.name_on_disk} data in {root}. Run {cls.__name__}.build() first."
            )
        with np.load(root / "stats.npz") as stats:
            meta = json.loads((root / "meta.json").read_text())
            return cls(
                name,
                np.load(root / f"X_{name}.npy", mmap_mode="r"),
                np.load(root / f"y_{name}.npy"),
                stats["mean"],
                stats["inv_std"],
                meta,
            )

    @classmethod
    @abstractmethod
    def build(cls, dataset_path: Path, out_dir: Path, seed: int = 0) -> None:
        """Turn the raw source files into X_*.npy, y_*.npy, stats.npz and meta.json (written last)."""
        raise NotImplementedError

    @classmethod
    def processed_dir(cls) -> Path:
        """Default folder for this source's processed files."""
        return get_base_path() / "dataset" / "processed" / cls.name_on_disk

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.meta["n_features"]

    @property
    def row_nbytes(self) -> int:
        """Bytes per row of X, for sizing batches against free device memory."""
        return self.n_features * self.x.itemsize

    def batches(
        self, size: int, rng: np.random.Generator | None = None
    ) -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Yield (x, y) slices. With rng, the block order is shuffled; without, it is fixed."""
        starts = np.arange(0, len(self), size)
        if rng is not None:
            rng.shuffle(starts)
        for s in starts:
            yield np.asarray(self.x[s : s + size]), self.y[s : s + size]

    def subset(self, n: int) -> Self:
        """The first n rows as a new dataset. Rows are pre-shuffled, so this is a fair sample."""
        return type(self)(self.name, self.x[:n], self.y[:n], self.mean, self.inv_std, self.meta)

    @classmethod
    def is_built(cls, root: Path | None = None) -> bool:
        """True once build() has finished for this source."""
        root = cls.processed_dir() if root is None else Path(root)
        return (root / "stats.npz").exists() and (root / "meta.json").exists()
