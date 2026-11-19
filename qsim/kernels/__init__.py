"""CUDA C source files for the GPU backend.

The .cu files in this folder are plain text. They are compiled at run time by CuPy
(through NVRTC), so no separate nvcc build step is needed.
"""

from importlib.resources import files

_REAL_TYPES = {"complex64": "float", "complex128": "double"}


def load_source(name: str) -> str:
    """Return the text of a kernel file in this package, e.g. load_source("gates.cu")."""
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


def build_source(name: str, dtype_name: str) -> str:
    """Full source for one kernel file at one precision: typedef + common header + file.

    dtype_name is "complex64" or "complex128".
    """
    if dtype_name not in _REAL_TYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}; use complex64 or complex128")
    header = f"typedef {_REAL_TYPES[dtype_name]} real_t;\n"
    return header + load_source("common.cuh") + "\n" + load_source(name)
