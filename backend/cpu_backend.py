from base import Backend, Array
import numpy as np
import threadpoolctl
import psutil

class CPUBackend(Backend):
    def __init__(self):
        self.xp = np
        self.name = "CPU (NumPy)"
        self.dtype = np.float32
        self.is_gpu = False

    def to_device(self, numpy_array: Array) -> Array:
        return np.ascontiguousarray(numpy_array, dtype=self.dtype)

    def to_host(self, backend_array: Array) -> Array:
        return np.asarray(backend_array, dtype=self.dtype)

    def to_scalar(self, backend_array: Array) -> float:
        return float(backend_array)

    def synchronize(self) -> None:
        pass

    def free_memory(self) -> None:
        pass #No GPU memory to free in CPU backend

    def set_cpu_threads(self, num_threads: int):
        return threadpoolctl.threadpool_limits(limits=num_threads, user_api='blas')

    def memory_info(self) -> tuple[int, int]:
        return (psutil.virtual_memory().available, psutil.virtual_memory().total)

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return np.zeros(shape, dtype=self.dtype)

    def from_rng(self, rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.01) -> Array:
        return (scale * rng.standard_normal(size=shape)).astype(self.dtype)

    def matmul(self, a: Array, b: Array) -> Array:
        return np.matmul(a, b)

    def add(self, a: Array, b: Array | float) -> Array:
        return np.add(a, b)

    def multiply(self, a: Array, b: Array | float) -> Array:
        return np.multiply(a, b)

    def sigmoid(self, a: Array) -> Array:
        return 1 / (1 + np.exp(-np.clip(a, -88, 88)))

    def log1pexp(self, a: Array) -> Array:
        """Compute log(1 + exp(a)) in a numerically stable way."""
        return np.logaddexp(a, 0)

    def sum (self, a: Array, axis: int | None = None) -> Array:
        return np.sum(a, axis=axis)

    def mean(self, a: Array, axis: int | None = None) -> Array:
        return np.mean(a, axis=axis)

    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        return np.clip(a, min_val, max_val)

    def transpose(self, a: Array) -> Array:
        return np.transpose(a)

    def axpy_(self, alpha: Array, x: Array, y: Array) -> None :
        np.add(y, alpha * x, out=y)
        

    def subtract(self, a: Array, b: Array | float) -> Array:
        return np.subtract(a, b)
    