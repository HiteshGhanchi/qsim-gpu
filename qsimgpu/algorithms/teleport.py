"""Quantum teleportation: move an unknown qubit state from qubit 0 to qubit 2.

Only two classical bits travel; the quantum state itself is never copied (it can't be).
  1. Qubits 1 and 2 share a Bell pair.
  2. Qubit 0 (holding the state) is entangled with qubit 1 (CX, then H on qubit 0).
  3. Qubits 0 and 1 are measured into clbits 0 and 1.
  4. Qubit 2 is corrected using those bits: X if clbit 1 = 1, Z if clbit 0 = 1.
Afterwards qubit 2 holds the original state, whichever of the four outcomes occurred.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qsimgpu.backends.base import Backend
from qsimgpu.circuit import Circuit
from qsimgpu.execution import RngLike


def preparation_matrix(psi: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """A unitary whose first column is psi, so it maps |0> to psi."""
    a, b = np.asarray(psi, dtype=np.complex128) / np.linalg.norm(psi)
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]])


def teleport_circuit(psi: NDArray[np.complexfloating]) -> Circuit:
    c = Circuit(3, n_clbits=2)
    c.unitary(preparation_matrix(psi), [0])   # the state to send
    c.h(1).cx(1, 2)                           # Bell pair on qubits 1 and 2
    c.cx(0, 1).h(0)                           # Bell-basis rotation of qubits 0 and 1
    c.measure(0, 0).measure(1, 1)
    c.x(2).c_if([1], 1)                       # corrections from the two classical bits
    c.z(2).c_if([0], 1)
    return c


def teleport_and_check(backend: Backend, psi: NDArray[np.complexfloating],
                       rng: RngLike = None) -> float:
    """Teleport psi and return the fidelity |<psi|qubit 2>|^2 (1 = perfect)."""
    backend.reset()
    m0, m1 = backend.run(teleport_circuit(psi), rng=rng)
    sv = backend.statevector()
    # qubits 0 and 1 are now the definite basis states m0, m1: read qubit 2's two amplitudes
    out = np.array([sv[m0 | (m1 << 1)], sv[m0 | (m1 << 1) | 4]], dtype=np.complex128)
    target = np.asarray(psi, dtype=np.complex128) / np.linalg.norm(psi)
    return float(abs(np.vdot(target, out)) ** 2)
