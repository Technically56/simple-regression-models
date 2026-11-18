from contextlib import AbstractContextManager, nullcontext

import cupy as cp
import numpy as np

from base import Array, BaseBackend


class GPUBackend(BaseBackend):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        cp.cuda.Device(device_id).use()
        self.xp = cp
        self.name = f"GPU (CuPy) - {cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()}"
        self.dtype = np.float32
        self.is_gpu = True

    def to_device(self, numpy_array: Array) -> Array:
        return cp.asarray(numpy_array, dtype=self.dtype)

    def to_host(self, backend_array: Array) -> np.ndarray:
        return cp.asnumpy(backend_array).astype(self.dtype, copy=False)

    def to_scalar(self, backend_array: Array) -> float:
        return float(backend_array)  # copies to host, so the CPU waits for the GPU

    def synchronize(self) -> None:
        cp.cuda.Device(self.device_id).synchronize()

    def free_memory(self) -> None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def set_cpu_threads(self, num_threads: int) -> AbstractContextManager:
        return nullcontext()  # BLAS threads don't apply to GPU work

    def memory_info(self) -> tuple[int, int]:
        free, total = cp.cuda.runtime.memGetInfo()
        return (free, total)

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return cp.zeros(shape, dtype=self.dtype)

    def from_rng(
        self, rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.01
    ) -> Array:
        # Draw on the CPU with NumPy so CPU and GPU runs start from identical weights.
        return cp.asarray((scale * rng.standard_normal(size=shape)).astype(self.dtype))

    def matmul(self, a: Array, b: Array) -> Array:
        return cp.matmul(a, b)

    def add(self, a: Array, b: Array | float) -> Array:
        return cp.add(a, b)

    def multiply(self, a: Array, b: Array | float) -> Array:
        return cp.multiply(a, b)

    def sigmoid(self, a: Array) -> Array:
        return 1 / (1 + cp.exp(-cp.clip(a, -88, 88)))

    def log1pexp(self, a: Array) -> Array:
        """Compute log(1 + exp(a)) in a numerically stable way."""
        return cp.logaddexp(a, 0)

    def sum(self, a: Array, axis: int | None = None) -> Array:
        return cp.sum(a, axis=axis)

    def mean(self, a: Array, axis: int | None = None) -> Array:
        return cp.mean(a, axis=axis)

    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        return cp.clip(a, min_val, max_val)

    def transpose(self, a: Array) -> Array:
        return cp.transpose(a)

    def axpy_(self, alpha: float, x: Array, y: Array) -> None:
        cp.add(y, alpha * x, out=y)

    def subtract(self, a: Array, b: Array | float) -> Array:
        return cp.subtract(a, b)

    def log(self, a: Array) -> Array:
        return cp.log(a)
