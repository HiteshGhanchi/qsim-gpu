"""CUDA backend: the statevector lives in GPU memory as one CuPy array of 2^n amplitudes.

Kernels are CUDA C files in qsim/kernels/, compiled at run time with NVRTC through
cupy.RawModule, once per precision. Every gate is one streaming pass over the state:
each GPU thread owns an independent group of amplitudes, so there are no races and no
synchronisation between threads.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qsim.backends.base import check_basis_state
from qsim.circuit import Circuit, Gate
from qsim.kernels import build_source

BLOCK = 256           # threads per block
MAX_BLOCKS = 65535    # grid-stride loops cover any remaining work


@cache
def _module(file: str, dtype_name: str) -> Any:
    import cupy as cp

    return cp.RawModule(code=build_source(file, dtype_name), options=("-std=c++17",))


def _mask(qubits: tuple[int, ...]) -> int:
    m = 0
    for q in qubits:
        m |= 1 << q
    return m


class CudaBackend:
    """Statevector simulator on an NVIDIA GPU (complex64 by default, complex128 optional)."""

    max_host_qubits = 26  # statevector()/probabilities() copy 2^n values to the host

    def __init__(self, n_qubits: int, dtype: type | np.dtype[Any] = np.complex64) -> None:
        import cupy as cp

        if not 1 <= n_qubits <= 34:
            raise ValueError("CudaBackend supports 1..34 qubits")
        self._cp = cp
        self.n_qubits = n_qubits
        self.dtype: np.dtype[Any] = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("dtype must be complex64 or complex128")
        self._psi = cp.zeros(1 << n_qubits, dtype=self.dtype)
        gates_mod = _module("gates.cu", self.dtype.name)
        self._k_apply_1q = gates_mod.get_function("apply_1q")
        self.reset()

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def _grid(n_work: int) -> tuple[int]:
        return (max(1, min(MAX_BLOCKS, (n_work + BLOCK - 1) // BLOCK)),)

    def _scalar(self, z: complex) -> np.complexfloating[Any, Any]:
        return self.dtype.type(z)  # type: ignore[no-any-return]

    def _check_qubits(self, gate: Gate) -> None:
        bad = [q for q in gate.qubits if q >= self.n_qubits]
        if bad:
            raise ValueError(f"{gate.name}: qubit(s) {bad} out of range")

    # --- state -----------------------------------------------------------------
    def reset(self, basis_state: int = 0) -> None:
        check_basis_state(basis_state, self.n_qubits)
        self._psi.fill(0)
        self._psi[basis_state] = 1

    def set_state(self, amplitudes: NDArray[np.complexfloating[Any, Any]]) -> None:
        """Load an arbitrary state from the host (for tests)."""
        a = np.asarray(amplitudes, dtype=self.dtype).reshape(-1)
        if a.size != 1 << self.n_qubits:
            raise ValueError("wrong number of amplitudes")
        self._psi[...] = self._cp.asarray(a)

    def _check_host_size(self) -> None:
        if self.n_qubits > self.max_host_qubits:
            raise ValueError(f"host copy refused for {self.n_qubits} qubits "
                             f"(limit {self.max_host_qubits})")

    def statevector(self) -> NDArray[Any]:
        self._check_host_size()
        return self._psi.get()  # type: ignore[no-any-return]

    # --- gates -----------------------------------------------------------------
    def apply_1q_matrix(self, m: NDArray[np.complex128], target: int,
                        controls: tuple[int, ...] = ()) -> None:
        """General 2x2 kernel: one thread per amplitude pair whose controls are all 1."""
        n_work = 1 << (self.n_qubits - 1 - len(controls))
        args = (self._psi, np.int64(n_work), np.int32(target),
                np.int64(_mask((target, *controls))), np.int64(_mask(controls)),
                self._scalar(m[0, 0]), self._scalar(m[0, 1]),
                self._scalar(m[1, 0]), self._scalar(m[1, 1]))
        self._k_apply_1q(self._grid(n_work), (BLOCK,), args)

    def apply(self, gate: Gate) -> None:
        self._check_qubits(gate)
        if len(gate.targets) == 1:
            self.apply_1q_matrix(gate.target_matrix(), gate.targets[0], gate.controls)
        else:
            raise NotImplementedError(f"{gate.name} on {len(gate.targets)} targets")

    def run(self, circuit: Circuit) -> None:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError(f"circuit has {circuit.n_qubits} qubits, backend {self.n_qubits}")
        for g in circuit:
            self.apply(g)

    def synchronize(self) -> None:
        self._cp.cuda.Device().synchronize()
