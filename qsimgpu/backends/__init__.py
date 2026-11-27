"""Simulator backends. CudaBackend is imported lazily because it needs CuPy and a GPU."""

from qsimgpu.backends.base import Backend
from qsimgpu.backends.numpy_backend import NumpyBackend

__all__ = ["Backend", "NumpyBackend"]
