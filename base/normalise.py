import numpy as np

from .base_backend import Array, BaseBackend


def normalise(
    backend: BaseBackend,
    x: np.ndarray,
    mean: Array,
    inv_std: Array,
    clip: float | None = None,
) -> Array:
    z = backend.multiply(backend.subtract(backend.to_device(x), mean), inv_std)
    return z if clip is None else backend.clip(z, -clip, clip)
