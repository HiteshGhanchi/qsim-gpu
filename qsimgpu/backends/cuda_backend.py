"""CUDA backend: the statevector lives in GPU memory as one CuPy array of 2^n amplitudes.

Kernels are CUDA C files in qsimgpu/kernels/, compiled at run time with NVRTC through
cupy.RawModule, once per precision. Every gate is one streaming pass over the state:
each GPU thread owns an independent group of amplitudes, so there are no races and no
synchronisation between threads.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qsimgpu.backends.base import (
    StepCallback,
    check_basis_state,
    check_marginal_qubits,
    pauli_masks,
)
from qsimgpu.circuit import Circuit, Gate
from qsimgpu.kernels import build_source

BLOCK = 256           # threads per block (also the reduction width in measure/expect.cu)
MAX_BLOCKS = 65535    # grid-stride loops cover any remaining work
REDUCE_BLOCKS = 1024  # blocks for reductions: 1024 partial sums, added on the host
MAX_K = 5             # largest multi-target matrix (kernels apply_k1 .. apply_k5)
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

    def __init__(self, n_qubits: int, dtype: type | np.dtype[Any] = np.complex64,
                 specialized: bool = True) -> None:
        """specialized=False routes every gate through the general kernels (the baseline)."""
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
        self._k_apply_k = {k: g.get_function(f"apply_k{k}") for k in range(1, MAX_K + 1)}
        self._k_phase = g.get_function("apply_phase")
        self._k_diag = g.get_function("apply_diag")
        self._k_x = g.get_function("apply_x")
        self._k_swap = g.get_function("apply_swap")
        self.specialized = specialized
        self._k_chunk_sums = m.get_function("chunk_prob_sums")
        self._k_locate = m.get_function("locate_in_chunk")
        self._k_marginal = m.get_function("marginal_probs")
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
            raise ValueError(f"multi-target kernels support 1..{MAX_K} targets, got {k}")
        n_work = 1 << (self.n_qubits - k - len(controls))
        pack = 0
        for i, t in enumerate(targets):
            pack |= t << (6 * i)
        mat = self._cp.asarray(np.ascontiguousarray(m, dtype=self.dtype))
        args = (self._psi, np.int64(n_work), np.int64(pack),
                np.int64(_mask(targets + controls)), np.int64(_mask(controls)), mat)
        self._k_apply_k[k](self._grid(n_work), (BLOCK,), args)

    def _apply_specialized(self, gate: Gate) -> bool:
        """Use a family-specific kernel if one fits. Returns False to fall back."""
        ctrls = gate.controls
        if gate.name == "swap":
            a, b = gate.targets
            n_work = 1 << (self.n_qubits - 2 - len(ctrls))
            self._k_swap(self._grid(n_work), (BLOCK,),
                         (self._psi, np.int64(n_work), np.int32(a), np.int32(b),
                          np.int64(_mask((a, b, *ctrls))), np.int64(_mask(ctrls))))
            return True
        if len(gate.targets) != 1 or gate.name == "unitary":
            return False
        (t,) = gate.targets
        if gate.kind == "permutation":  # x, cx, ccx, multi-controlled x
            n_work = 1 << (self.n_qubits - 1 - len(ctrls))
            self._k_x(self._grid(n_work), (BLOCK,),
                      (self._psi, np.int64(n_work), np.int32(t),
                       np.int64(_mask((t, *ctrls))), np.int64(_mask(ctrls))))
            return True
        if gate.kind == "diagonal":
            m = gate.target_matrix()
            d0, d1 = complex(m[0, 0]), complex(m[1, 1])
            if d0 == 1:  # diag(1, phase): touch only amplitudes with target and controls = 1
                fixed = _mask((t, *ctrls))
                n_work = 1 << (self.n_qubits - 1 - len(ctrls))
                self._k_phase(self._grid(n_work), (BLOCK,),
                              (self._psi, np.int64(n_work), np.int64(fixed), np.int64(fixed),
                               self._scalar(d1)))
            else:  # e.g. rz: both entries change
                n_work = 1 << (self.n_qubits - len(ctrls))
                self._k_diag(self._grid(n_work), (BLOCK,),
                             (self._psi, np.int64(n_work), np.int32(t),
                              np.int64(_mask(ctrls)), np.int64(_mask(ctrls)),
                              self._scalar(d0), self._scalar(d1)))
            return True
        return False

    def apply(self, gate: Gate) -> None:
        self._check_qubits(gate)
        if self.specialized and self._apply_specialized(gate):
            return
        if len(gate.targets) == 1:
            self.apply_1q_matrix(gate.target_matrix(), gate.targets[0], gate.controls)
        else:
            self.apply_kq_matrix(gate.target_matrix(), gate.targets, gate.controls)

    def run(self, circuit: Circuit, on_step: StepCallback | None = None) -> None:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError(f"circuit has {circuit.n_qubits} qubits, backend {self.n_qubits}")
        for i, g in enumerate(circuit):
            self.apply(g)
            if on_step is not None:
                on_step(i, g, self)

    def marginal_probabilities(self, qubits: Sequence[int]) -> NDArray[np.float64]:
        """Marginal distribution of up to 12 qubits, reduced on the GPU (no full host copy)."""
        cp = self._cp
        qs = check_marginal_qubits(qubits, self.n_qubits)
        k = len(qs)
        out = cp.zeros(1 << k, dtype=cp.float64)
        n_amp = 1 << self.n_qubits
        blocks = max(1, min(REDUCE_BLOCKS, (n_amp + BLOCK - 1) // BLOCK))
        self._k_marginal((blocks,), (BLOCK,),
                         (self._psi, np.int64(n_amp), np.int32(k),
                          cp.asarray(np.array(qs, dtype=np.int32)), out),
                         shared_mem=8 * (1 << k))
        return out.get()  # type: ignore[no-any-return]

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
