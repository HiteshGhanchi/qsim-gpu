"""Cross-check the NumPy oracle against Qiskit Aer on random circuits (no reordering)."""

import numpy as np
import pytest

from qsim.backends import NumpyBackend
from qsim.circuit import Circuit
from tests.circuits import random_circuit
from tests.tolerances import ATOL_COMPLEX128

pytest.importorskip("qiskit_aer")
from tests.qiskit_bridge import aer_statevector  # noqa: E402


def oracle(c: Circuit) -> np.ndarray:
    b = NumpyBackend(c.n_qubits)
    b.run(c)
    return b.statevector()


def test_endianness_probe_matches_aer() -> None:
    c = Circuit(3).x(0)
    np.testing.assert_array_equal(np.round(aer_statevector(c), 12), oracle(c))


@pytest.mark.parametrize("seed", range(20))
def test_random_circuits_match_aer_exactly(seed: int) -> None:
    n = 1 + seed % 12
    c = random_circuit(n, 80, seed)
    np.testing.assert_allclose(oracle(c), aer_statevector(c), atol=ATOL_COMPLEX128)


def test_unitary_gate_matches_aer() -> None:
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8)))
    c = Circuit(4).h(0).h(3).unitary(q, [3, 0, 2])
    np.testing.assert_allclose(oracle(c), aer_statevector(c), atol=ATOL_COMPLEX128)
