"""NumPy reference backend: the oracle every other backend is judged against.

Goal: obviously correct, not fast.
- The state is an ndarray of shape (2,)*n in complex128 (more precise than the GPU's complex64).
- Qubit k lives on axis n-1-k, so the flat C-order index equals the little-endian integer
  (axis 0 is the most significant bit = qubit n-1).
- Every gate, of any kind, goes through one general routine: select the slice where all
  controls are 1, then contract the target axes with the gate matrix (np.tensordot).
- Pauli expectations are computed by literally applying the Pauli matrices and taking an
  inner product. This is deliberately a different method from the GPU's bit-mask formula,
  so the two can catch each other's mistakes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qsimgpu import gates
from qsimgpu.backends.base import check_basis_state, check_pauli
from qsimgpu.circuit import Circuit, Gate

_PAULI = {
    "X": gates.base_matrix("x"),
    "Y": gates.base_matrix("y"),
    "Z": gates.base_matrix("z"),
}


class NumpyBackend:
    """Statevector simulator in NumPy (complex128)."""

    def __init__(self, n_qubits: int) -> None:
        if not 1 <= n_qubits <= 28:
            raise ValueError("NumpyBackend supports 1..28 qubits")
        self.n_qubits = n_qubits
        self._psi: NDArray[np.complex128] = np.zeros((2,) * n_qubits, dtype=np.complex128)
        self.reset()

    # --- state ---------------------------------------------------------------
    def reset(self, basis_state: int = 0) -> None:
        check_basis_state(basis_state, self.n_qubits)
        self._psi.fill(0)
        self._psi.reshape(-1)[basis_state] = 1.0

    def set_state(self, amplitudes: NDArray[np.complexfloating]) -> None:
        """Load an arbitrary normalised state (for tests)."""
        a = np.asarray(amplitudes, dtype=np.complex128).reshape((2,) * self.n_qubits)
        self._psi[...] = a

    def statevector(self) -> NDArray[np.complex128]:
        return self._psi.reshape(-1).copy()

    def probabilities(self) -> NDArray[np.float64]:
        return np.abs(self._psi.reshape(-1)) ** 2

    def amplitude_norm(self) -> float:
        return float(np.linalg.norm(self._psi))

    # --- gates -----------------------------------------------------------------
    def _axis(self, qubit: int) -> int:
        return self.n_qubits - 1 - qubit

    def apply_matrix(self, matrix: NDArray[np.complex128], targets: tuple[int, ...],
                     controls: tuple[int, ...] = ()) -> None:
        """Apply a 2^k x 2^k matrix to `targets`, only where every control qubit is 1."""
        n, k = self.n_qubits, len(targets)
        index: list[int | slice] = [slice(None)] * n
        for c in controls:
            index[self._axis(c)] = 1
        view = self._psi[tuple(index)]  # control axes removed; the rest keep their order

        remaining = [ax for ax in range(n) if ax not in {self._axis(c) for c in controls}]
        target_axes = [remaining.index(self._axis(t)) for t in targets]
        # Reshaping the matrix to (2,)*2k orders its axes most-significant first, i.e.
        # targets[k-1] ... targets[0] for the output half and again for the input half.
        msb_first = target_axes[::-1]
        m = matrix.reshape((2,) * (2 * k))
        out = np.tensordot(m, view, axes=(list(range(k, 2 * k)), msb_first))
        out = np.moveaxis(out, list(range(k)), msb_first)
        self._psi[tuple(index)] = out

    def apply(self, gate: Gate) -> None:
        bad = [q for q in gate.qubits if q >= self.n_qubits]
        if bad:
            raise ValueError(f"{gate.name}: qubit(s) {bad} out of range")
        self.apply_matrix(gate.target_matrix(), gate.targets, gate.controls)

    def run(self, circuit: Circuit) -> None:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError(f"circuit has {circuit.n_qubits} qubits, backend {self.n_qubits}")
        for g in circuit:
            self.apply(g)

    # --- measurement -------------------------------------------------------------
    def sample(self, shots: int, seed: int | None = None) -> dict[int, int]:
        if shots < 0:
            raise ValueError("shots must be >= 0")
        p = self.probabilities()
        p = p / p.sum()
        rng = np.random.default_rng(seed)
        draws = rng.choice(p.size, size=shots, p=p)
        values, counts = np.unique(draws, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=True)}

    def pauli_value(self, pauli: str) -> complex:
        """<psi|P|psi> as a complex number. For a Pauli string it must come out real."""
        check_pauli(pauli, self.n_qubits)
        saved = self._psi.copy()
        try:
            for pos, ch in enumerate(pauli):
                if ch != "I":
                    self.apply_matrix(_PAULI[ch], (self.n_qubits - 1 - pos,))
            return complex(np.vdot(saved, self._psi))  # <psi| P |psi>
        finally:
            self._psi[...] = saved

    def expectation_pauli(self, pauli: str, coeff: complex = 1.0) -> float:
        return float((coeff * self.pauli_value(pauli)).real)

    def expectation_hamiltonian(self, terms: list[tuple[float, str]]) -> float:
        return float(sum(self.expectation_pauli(p, c) for c, p in terms))
