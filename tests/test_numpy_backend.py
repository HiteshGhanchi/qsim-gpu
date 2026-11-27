"""NumPy reference backend: known states, bit order, invariants, measurement, Pauli values."""

import math

import numpy as np
import pytest

from qsimgpu.backends import NumpyBackend
from qsimgpu.circuit import Circuit
from tests.circuits import random_circuit, random_pauli, random_state
from tests.tolerances import ATOL_COMPLEX128, NORM_ATOL_COMPLEX128

S2 = 1 / math.sqrt(2)


def run(c: Circuit) -> NumpyBackend:
    b = NumpyBackend(c.n_qubits)
    b.run(c)
    return b


def test_bit_order_x_on_qubit0_gives_index_1() -> None:
    sv = run(Circuit(3).x(0)).statevector()
    assert sv[1] == 1 and np.count_nonzero(sv) == 1


def test_bit_order_x_on_top_qubit() -> None:
    sv = run(Circuit(3).x(2)).statevector()
    assert sv[4] == 1


def test_bell_state() -> None:
    sv = run(Circuit(2).h(0).cx(0, 1)).statevector()
    np.testing.assert_allclose(sv, [S2, 0, 0, S2], atol=1e-15)


@pytest.mark.parametrize("n", [3, 10, 20])
def test_ghz(n: int) -> None:
    c = Circuit(n).h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    sv = run(c).statevector()
    assert np.flatnonzero(np.abs(sv) > 1e-12).tolist() == [0, (1 << n) - 1]
    np.testing.assert_allclose(abs(sv[0]), S2)


def test_h_h_returns_to_zero() -> None:
    sv = run(Circuit(1).h(0).h(0)).statevector()
    np.testing.assert_allclose(sv, [1, 0], atol=1e-15)


def test_h_h_with_measurement_in_between_is_fifty_fifty() -> None:
    # Measure after the first H, collapse to the outcome, then apply the second H.
    b = NumpyBackend(1)
    rng = np.random.default_rng(1)
    outcomes = []
    for _ in range(2000):
        b.reset()
        b.run(Circuit(1).h(0))
        (collapsed,) = b.sample(1, seed=int(rng.integers(1 << 31))).keys()
        b.reset(collapsed)
        b.run(Circuit(1).h(0))
        outcomes.extend(b.sample(1, seed=int(rng.integers(1 << 31))).keys())
    assert 0.45 < np.mean(outcomes) < 0.55


def test_controlled_rz_is_not_controlled_p() -> None:
    # Uncontrolled, Rz and P differ only by a global phase; controlled, the phase is relative.
    a = run(Circuit(2).h(0).h(1).add("rz", [1], [0], [0.8])).statevector()
    b = run(Circuit(2).h(0).h(1).cp(0.8, 0, 1)).statevector()
    overlap = abs(np.vdot(a, b))
    assert overlap < 1 - 1e-3


@pytest.mark.parametrize("seed", range(10))
def test_norm_preserved_and_inverse_returns_to_start(seed: int) -> None:
    n = 1 + seed % 8
    c = random_circuit(n, 60, seed)
    b = NumpyBackend(n)
    for g in c:
        b.apply(g)
        assert abs(b.amplitude_norm() - 1) < NORM_ATOL_COMPLEX128
    b.run(c.inverse())
    expected = np.zeros(1 << n)
    expected[0] = 1
    np.testing.assert_allclose(b.statevector(), expected, atol=ATOL_COMPLEX128)


def test_unitary_gate_matches_named_gates() -> None:
    cx_matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    # targets (0, 1): bit 0 of the matrix index is qubit 0 = control, bit 1 = target.
    a = run(Circuit(3).h(0).unitary(cx_matrix, [0, 1])).statevector()
    b = run(Circuit(3).h(0).cx(0, 1)).statevector()
    np.testing.assert_allclose(a, b, atol=1e-15)


def test_sample_counts_follow_probabilities() -> None:
    b = run(Circuit(2).h(0).cx(0, 1))
    counts = b.sample(4000, seed=7)
    assert set(counts) == {0, 3}
    assert sum(counts.values()) == 4000
    assert abs(counts[0] / 4000 - 0.5) < 0.05


def test_pauli_order_leftmost_is_highest_qubit() -> None:
    b = run(Circuit(2).x(0))  # qubit 0 = 1, qubit 1 = 0
    assert b.expectation_pauli("IZ") == pytest.approx(-1)  # Z on qubit 0
    assert b.expectation_pauli("ZI") == pytest.approx(1)   # Z on qubit 1


def test_pauli_values_are_real_and_match_dense_matrix() -> None:
    n = 4
    b = NumpyBackend(n)
    pm = {"I": np.eye(2), "X": [[0, 1], [1, 0]], "Y": [[0, -1j], [1j, 0]], "Z": [[1, 0], [0, -1]]}
    for seed in range(20):
        psi = random_state(n, seed)
        pauli = random_pauli(n, 100 + seed)
        b.set_state(psi)
        value = b.pauli_value(pauli)
        dense = np.array([[1.0]])
        for ch in pauli:  # leftmost char = most significant = first Kronecker factor
            dense = np.kron(dense, np.asarray(pm[ch]))
        assert abs(value.imag) < 1e-12
        assert value.real == pytest.approx(np.vdot(psi, dense @ psi).real, abs=1e-12)


def test_expectation_hamiltonian_sums_terms() -> None:
    b = run(Circuit(2).h(0).cx(0, 1))
    terms = [(0.5, "ZZ"), (0.25, "XX"), (2.0, "II"), (1.0, "ZI")]
    assert b.expectation_hamiltonian(terms) == pytest.approx(0.5 + 0.25 + 2.0 + 0.0)
