from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

type Array = np.ndarray | cp.ndarray


class BaseBackend(ABC):
    """Abstract base class for math backends."""

    name: str
    is_gpu: bool
    dtype: type
    xp: ModuleType

    @abstractmethod
    def to_device(self, numpy_array: np.ndarray) -> Array:
        """Move a NumPy array to the backend's device (e.g., GPU for CuPy). Always uses dtype for the converted arrays."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def to_host(self, backend_array: Array) -> np.ndarray:
        """Move an array from the backend's device back to the host (CPU)."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def to_scalar(self, backend_array: Array) -> float:
        """Extract a scalar value from a backend array."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize the backend's device (e.g., wait for GPU computations to finish)."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def free_memory(self) -> None:
        """Free any cached memory on the backend's device (e.g., GPU)."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def set_cpu_threads(self, num_threads: int) -> AbstractContextManager:
        """Set the number of CPU threads for the backend (if applicable)."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def memory_info(self) -> tuple[int, int]:
        """Return a tuple of (free_memory, total_memory) in bytes for the backend's device."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Array:
        """Return a backend-specific array of zeros."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def from_rng(
        self, rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.01
    ) -> Array:
        """Generate a backend-specific random array using the provided NumPy random generator."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def matmul(self, a: Array, b: Array) -> Array:
        """Perform matrix multiplication using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def sigmoid(self, a: Array) -> Array:
        """Compute the sigmoid function using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def log1pexp(self, a: Array) -> Array:
        """Compute log(1 + exp(a)) in a numerically stable way using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def sum(self, a: Array, axis: int | None = None) -> Array:
        """Compute the sum of array elements over a given axis using the backend. Returns a 0-d array when axis is None, never a Python float."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def mean(self, a: Array, axis: int | None = None) -> Array:
        """Compute the mean of array elements over a given axis using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def clip(self, a: Array, min_val: float, max_val: float) -> Array:
        """Clip (limit) the values in an array using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def transpose(self, a: Array) -> Array:
        """Transpose the array using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def axpy_(self, alpha: float, x: Array, y: Array) -> None:
        """Perform the operation y := alpha * x + y in-place using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def add(self, a: Array, b: Array | float) -> Array:
        """Add two arrays using the backend"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def subtract(self, a: Array, b: Array | float) -> Array:
        """Subtract two arrays using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def multiply(self, a: Array, b: Array | float) -> Array:
        """Multiply two arrays using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def log(self, a: Array) -> Array:
        """Compute the natural logarithm of an array using the backend."""
        raise NotImplementedError("This method should be implemented by subclasses.")
