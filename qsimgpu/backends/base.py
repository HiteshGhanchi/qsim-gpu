"""The Backend interface every simulator implements, plus helpers they share.

Pauli strings: "XIZY" names one Pauli per qubit. The LEFTMOST character is the HIGHEST
qubit index, the same order used to write bit patterns (qubit 0 is the rightmost character).
So for "XIZY": qubit 3 = X, qubit 2 = I, qubit 1 = Z, qubit 0 = Y.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from qsimgpu.circuit import Circuit, Gate

PAULI_CHARS = frozenset("IXYZ")


@runtime_checkable
class Backend(Protocol):
    n_qubits: int

    def reset(self, basis_state: int = 0) -> None:
        """Set the state to one basis pattern (default |0...0>)."""
        ...

    def apply(self, gate: Gate) -> None:
        """Apply one gate in place."""
        ...

    def run(self, circuit: Circuit) -> None:
        """Apply all gates of the circuit in order, in place."""
        ...

    def statevector(self) -> np.ndarray:
        """Host copy of all 2^n amplitudes. Only sensible for n <= ~26."""
        ...

    def probabilities(self) -> np.ndarray:
        """Host copy of |amplitude|^2 for every pattern. Same size limit."""
        ...

    def sample(self, shots: int, seed: int | None = None) -> dict[int, int]:
        """Measure `shots` times without collapsing; returns {pattern: count}."""
        ...

    def expectation_pauli(self, pauli: str, coeff: complex = 1.0) -> float:
        """Exact <psi| coeff * P |psi> for one Pauli string (real part)."""
        ...

    def expectation_hamiltonian(self, terms: list[tuple[float, str]]) -> float:
        """Exact sum of coeff * <P> over (coeff, pauli) terms. No shot noise."""
        ...

    def amplitude_norm(self) -> float:
        """sqrt(sum |a|^2); should stay 1 after every gate."""
        ...


def check_pauli(pauli: str, n_qubits: int) -> None:
    if len(pauli) != n_qubits:
        raise ValueError(f"Pauli string {pauli!r} has {len(pauli)} chars, need {n_qubits}")
    if not set(pauli) <= PAULI_CHARS:
        raise ValueError(f"Pauli string {pauli!r} may only contain I, X, Y, Z")


def pauli_masks(pauli: str, n_qubits: int) -> tuple[int, int, int]:
    """(x_mask, z_mask, n_y) for a Pauli string.

    x_mask: bit q set where qubit q has X or Y (these flip the bit).
    z_mask: bit q set where qubit q has Z or Y (these give a sign from the bit).
    n_y:    number of Y's (each contributes a factor i).
    With these, P|i> = i^n_y * (-1)^popcount(i & z_mask) * |i ^ x_mask>.
    """
    check_pauli(pauli, n_qubits)
    x_mask = z_mask = n_y = 0
    for pos, ch in enumerate(pauli):
        q = n_qubits - 1 - pos
        if ch in "XY":
            x_mask |= 1 << q
        if ch in "ZY":
            z_mask |= 1 << q
        if ch == "Y":
            n_y += 1
    return x_mask, z_mask, n_y


def check_basis_state(basis_state: int, n_qubits: int) -> None:
    if not 0 <= basis_state < (1 << n_qubits):
        raise ValueError(f"basis state {basis_state} out of range for {n_qubits} qubits")
