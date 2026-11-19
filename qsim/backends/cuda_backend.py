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

from qsim.backends.base import check_basis_state, pauli_masks
from qsim.circuit import Circuit, Gate
from qsim.kernels import build_source

BLOCK = 256           # threads per block (also the reduction width in measure/expect.cu)
MAX_BLOCKS = 65535    # grid-stride loops cover any remaining work
REDUCE_BLOCKS = 1024  # blocks for reductions: 1024 partial sums, added on the host
MAX_K = 5             # largest multi-target matrix the apply_kq kernel accepts
CHUNK_LOG2 = 10       # sampling chunk = 1024 amplitudes


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
        g = _module("gates.cu", self.dtype.name)
        m = _module("measure.cu", self.dtype.name)
        e = _module("expect.cu", self.dtype.name)
        self._k_apply_1q = g.get_function("apply_1q")
        self._k_apply_kq = g.get_function("apply_kq")
        self._k_chunk_sums = m.get_function("chunk_prob_sums")
        self._k_locate = m.get_function("locate_in_chunk")
        self._k_pauli = e.get_function("pauli_partial")
        self.reset()

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def _grid(n_work: int) -> tuple[int]:
        return (max(1, min(MAX_BLOCKS, (n_work + BLOCK - 1) // BLOCK)),)

    def _scalar(self, z: complex) -> Any:
        return self.dtype.type(z)

    def _check_qubits(self, gate: Gate) -> None:
        bad = [q for q in gate.qubits if q >= self.n_qubits]
        if bad:
            raise ValueError(f"{gate.name}: qubit(s) {bad} out of range")

    def synchronize(self) -> None:
        self._cp.cuda.Device().synchronize()

    @property
    def state(self) -> Any:
        """The device array itself (no copy). For benchmarks and advanced use."""
        return self._psi

    # --- state -----------------------------------------------------------------
    def reset(self, basis_state: int = 0) -> None:
        check_basis_state(basis_state, self.n_qubits)
        self._psi.fill(0)
        self._psi[basis_state] = 1

    def set_state(self, amplitudes: NDArray[Any]) -> None:
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

    def probabilities(self) -> NDArray[np.float64]:
        self._check_host_size()
        p = self._cp.abs(self._psi) ** 2
        return p.get().astype(np.float64)  # type: ignore[no-any-return]

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

    def apply_kq_matrix(self, m: NDArray[np.complex128], targets: tuple[int, ...],
                        controls: tuple[int, ...] = ()) -> None:
        """General 2^k x 2^k kernel: one thread per group of 2^k amplitudes."""
        k = len(targets)
        if not 1 <= k <= MAX_K:
            raise ValueError(f"apply_kq supports 1..{MAX_K} targets, got {k}")
        n_work = 1 << (self.n_qubits - k - len(controls))
        pack = 0
        for i, t in enumerate(targets):
            pack |= t << (6 * i)
        mat = self._cp.asarray(np.ascontiguousarray(m, dtype=self.dtype))
        args = (self._psi, np.int64(n_work), np.int32(k), np.int64(pack),
                np.int64(_mask(targets + controls)), np.int64(_mask(controls)), mat)
        self._k_apply_kq(self._grid(n_work), (BLOCK,), args)

    def apply(self, gate: Gate) -> None:
        self._check_qubits(gate)
        if len(gate.targets) == 1:
            self.apply_1q_matrix(gate.target_matrix(), gate.targets[0], gate.controls)
        else:
            self.apply_kq_matrix(gate.target_matrix(), gate.targets, gate.controls)

    def run(self, circuit: Circuit) -> None:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError(f"circuit has {circuit.n_qubits} qubits, backend {self.n_qubits}")
        for g in circuit:
            self.apply(g)

    # --- measurement -------------------------------------------------------------
    def sample(self, shots: int, seed: int | None = None) -> dict[int, int]:
        """Two-pass chunked sampling; extra memory is one double per 1024 amplitudes."""
        cp = self._cp
        if shots < 0:
            raise ValueError("shots must be >= 0")
        if shots == 0:
            return {}
        chunk_log2 = min(CHUNK_LOG2, self.n_qubits)
        n_chunks = 1 << (self.n_qubits - chunk_log2)
        sums = cp.empty(n_chunks, dtype=cp.float64)
        self._k_chunk_sums((min(n_chunks, MAX_BLOCKS),), (BLOCK,),
                           (self._psi, np.int32(chunk_log2), np.int64(n_chunks), sums))
        cum = cp.cumsum(sums)
        total = float(cum[-1])
        u = cp.asarray(np.random.default_rng(seed).random(shots) * total)
        ids = cp.minimum(cp.searchsorted(cum, u, side="right"), n_chunks - 1).astype(cp.int64)
        residual = u - (cum[ids] - sums[ids])
        picks = cp.empty(shots, dtype=cp.int64)
        self._k_locate(self._grid(shots), (BLOCK,),
                       (self._psi, np.int32(chunk_log2), ids, residual, picks, np.int64(shots)))
        values, counts = np.unique(picks.get(), return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=True)}

    def sample_cumsum(self, shots: int, seed: int | None = None) -> dict[int, int]:
        """Simple alternative: cumulative sum over all 2^n probabilities (float64).

        Allocates 8 bytes per amplitude, more than the complex64 state itself, so it is
        only kept for comparison with sample() at small n.
        """
        cp = self._cp
        self._check_host_size()
        cum = cp.cumsum(cp.abs(self._psi).astype(cp.float64) ** 2)
        u = cp.asarray(np.random.default_rng(seed).random(shots) * float(cum[-1]))
        idx = cp.minimum(cp.searchsorted(cum, u, side="right"), cum.size - 1)
        values, counts = np.unique(idx.get(), return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=True)}

    # --- expectation values ---------------------------------------------------------
    def _pauli_sum(self, x_mask: int, z_mask: int) -> complex:
        cp = self._cp
        n_amp = 1 << self.n_qubits
        blocks = max(1, min(REDUCE_BLOCKS, (n_amp + BLOCK - 1) // BLOCK))
        re = cp.empty(blocks, dtype=cp.float64)
        im = cp.empty(blocks, dtype=cp.float64)
        self._k_pauli((blocks,), (BLOCK,),
                      (self._psi, np.int64(n_amp), np.int64(x_mask), np.int64(z_mask), re, im))
        return complex(float(re.sum()), float(im.sum()))

    def pauli_value(self, pauli: str) -> complex:
        """<psi|P|psi> as a complex number. For a Pauli string it must come out real."""
        x_mask, z_mask, n_y = pauli_masks(pauli, self.n_qubits)
        return complex(1j**n_y * self._pauli_sum(x_mask, z_mask))

    def expectation_pauli(self, pauli: str, coeff: complex = 1.0) -> float:
        return float((coeff * self.pauli_value(pauli)).real)

    def expectation_hamiltonian(self, terms: list[tuple[float, str]]) -> float:
        return float(sum(self.expectation_pauli(p, c) for c, p in terms))

    def amplitude_norm(self) -> float:
        return float(np.sqrt(self._pauli_sum(0, 0).real))
