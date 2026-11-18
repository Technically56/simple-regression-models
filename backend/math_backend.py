import logging
from os import environ

from backend import CPUBackend
from base import perf
from base.base_backend import BaseBackend

log = logging.getLogger(__name__)


def get_backend() -> BaseBackend:
    """Return the backend chosen by MATH_BACKEND (numpy or cupy) and CUDA_DEVICE."""
    if environ.get("MATH_BACKEND", "numpy").lower().strip() == "cupy":
        try:
            import cupy as cp
        except ImportError:
            raise RuntimeError(
                "CuPy is not installed. Please install it to use the cupy backend."
            )
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(
                f"CuPy is installed but no working CUDA device was found: {e}"
            )

        from backend.gpu_backend import (
            GPUBackend,  # imported here so CPU-only machines never import cupy
        )

        device_id = int(environ.get("CUDA_DEVICE", "0"))
        backend = GPUBackend(device_id)
        log.info(
            "CUDA devices found: %d. Using device %d: %s",
            device_count,
            device_id,
            backend.name,
        )
    else:
        backend = CPUBackend()
        log.info("Using CPU backend (NumPy).")

    perf("device", backend=backend.name, is_gpu=backend.is_gpu)
    return backend
