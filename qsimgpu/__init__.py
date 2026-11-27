"""qsimgpu: a statevector quantum circuit simulator with a NumPy reference and a CUDA backend.

Public API (everything else is internal and may change between versions):

    from qsimgpu import Circuit, Gate, NumpyBackend, CudaBackend
    from qsimgpu import from_qasm2, to_qasm2, run_shots, fuse
    from qsimgpu import views          # marginals, Bloch vectors, step traces
    from qsimgpu import algorithms     # grover, qft, shor15, teleport, bitflip, vqe

CudaBackend is loaded on first use, so `import qsimgpu` needs only NumPy; CuPy and an NVIDIA GPU
are needed only when a CudaBackend is created.

Conventions used everywhere in this package:
- Little-endian qubit order: qubit 0 is the least significant bit of a state index.
- Pauli strings are written with the highest qubit first ("XIZY": Y on qubit 0).
- Gate definitions follow Qiskit.
- The CUDA backend defaults to complex64; the NumPy reference uses complex128.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qsimgpu import algorithms, views
from qsimgpu.backends import Backend, NumpyBackend
from qsimgpu.circuit import Circuit, Gate
from qsimgpu.execution import run_shots
from qsimgpu.passes.fusion import fuse
from qsimgpu.qasm import QasmError, QasmExportError, from_qasm2, to_qasm2

if TYPE_CHECKING:
    from qsimgpu.backends.cuda_backend import CudaBackend

__version__ = "0.2.0"
__all__ = [
    "Backend", "Circuit", "CudaBackend", "Gate", "NumpyBackend", "QasmError", "QasmExportError",
    "algorithms", "from_qasm2", "fuse", "run_shots", "to_qasm2", "views", "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "CudaBackend":  # import CuPy only when the GPU backend is actually used
        from qsimgpu.backends.cuda_backend import CudaBackend

        return CudaBackend
    raise AttributeError(f"module 'qsimgpu' has no attribute {name!r}")
