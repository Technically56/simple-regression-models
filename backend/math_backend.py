from base import Backend, Array

from os import environ
from pathlib import Path

def get_backend() -> Backend:
    """Return the default backend for the project."""
    if environ.get("MATH_BACKEND", "numpy") == "cupy":
        try:
            import cupy as cp
        except ImportError:
            raise RuntimeError("CuPy is not installed. Please install it to use the cupy backend.")

        from backend.cupy_backend import CuPyBackend
        return CuPyBackend()
    
    return CPUBackend()