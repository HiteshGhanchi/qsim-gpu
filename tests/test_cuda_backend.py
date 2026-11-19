"""CUDA backend vs the NumPy oracle."""

from __future__ import annotations

import numpy as np
import pytest

from qsim.backends import NumpyBackend
from qsim.circuit import Circuit, Gate
from tests.circuits import random_circuit, random_state
from tests.tolerances import ATOL_COMPLEX64, ATOL_COMPLEX128

pytestmark = pytest.mark.gpu
DTYPES = [(np.complex64, ATOL_COMPLEX64), (np.complex128, ATOL_COMPLEX128)]


def cuda(n: int, dtype: type = np.complex64):  # type: ignore[no-untyped-def]
    from qsim.backends.cuda_backend import CudaBackend

    return CudaBackend(n, dtype)


def oracle(c: Circuit) -> np.ndarray:
    b = NumpyBackend(c.n_qubits)
    b.run(c)
    return b.statevector()


def full_matrix(gate: Gate, n: int) -> np.ndarray:
    """Explicit 2^n x 2^n matrix built column by column from the bit definition (no tricks)."""
    m = gate.target_matrix()
    k = len(gate.targets)
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    for col in range(dim):
        if any(not (col >> c) & 1 for c in gate.controls):
            out[col, col] = 1
            continue
        sub_in = sum(((col >> t) & 1) << i for i, t in enumerate(gate.targets))
        base = col
        for t in gate.targets:
            base &= ~(1 << t)
        for sub_out in range(1 << k):
            row = base
            for i, t in enumerate(gate.targets):
                row |= ((sub_out >> i) & 1) << t
            out[row, col] += m[sub_out, sub_in]
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_single_gate_matches_explicit_matrix(n: int) -> None:
    rng = np.random.default_rng(n)
    for trial in range(30):
        name = str(rng.choice(["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p"]))
        qs = [int(q) for q in rng.permutation(n)]
        n_ctrl = int(rng.integers(0, n)) if n > 1 else 0
        params = (float(rng.uniform(-3, 3)),) if name in ("rx", "ry", "rz", "p") else ()
        g = Gate(name, (qs[0],), tuple(qs[1 : 1 + n_ctrl]), params)
        psi = random_state(n, trial)
        b = cuda(n, np.complex128)
        b.set_state(psi)
        b.apply(g)
        np.testing.assert_allclose(b.statevector(), full_matrix(g, n) @ psi, atol=1e-12)


@pytest.mark.parametrize(("dtype", "atol"), DTYPES)
@pytest.mark.parametrize("n", [1, 2, 5, 10, 14, 20])
def test_one_qubit_circuits_match_oracle(n: int, dtype: type, atol: float) -> None:
    for seed in range(3):
        c = random_circuit(n, 100, seed, only_one_qubit=True)
        b = cuda(n, dtype)
        b.run(c)
        np.testing.assert_allclose(b.statevector(), oracle(c), atol=atol)


def test_reset_to_basis_state() -> None:
    b = cuda(4)
    b.reset(5)
    sv = b.statevector()
    assert sv[5] == 1 and np.count_nonzero(sv) == 1


# --- phase 3: full gate set, measurement, expectation ----------------------------

@pytest.mark.parametrize(("dtype", "atol"), DTYPES)
@pytest.mark.parametrize("seed", range(16))
def test_random_full_circuits_match_oracle(seed: int, dtype: type, atol: float) -> None:
    n = 1 + seed % 14
    c = random_circuit(n, 100, seed)
    b = cuda(n, dtype)
    b.run(c)
    np.testing.assert_allclose(b.statevector(), oracle(c), atol=atol)


@pytest.mark.parametrize("n", [18, 20])
def test_large_random_circuits_match_oracle(n: int) -> None:
    c = random_circuit(n, 100, 77 + n)
    b = cuda(n)
    b.run(c)
    np.testing.assert_allclose(b.statevector(), oracle(c), atol=ATOL_COMPLEX64)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_multi_target_gates_match_explicit_matrix(n: int) -> None:
    rng = np.random.default_rng(10 + n)
    for trial in range(10):
        qs = [int(q) for q in rng.permutation(n)]
        k = int(rng.integers(1, min(n, 3) + 1))
        q, _ = np.linalg.qr(rng.standard_normal((1 << k, 1 << k))
                            + 1j * rng.standard_normal((1 << k, 1 << k)))
        gates_to_try = [Gate("unitary", tuple(qs[:k]), matrix=q)]
        if n >= 2:
            ctrls = tuple(qs[2 : 2 + int(rng.integers(0, n - 1))])
            gates_to_try.append(Gate("swap", (qs[0], qs[1]), ctrls))
        for g in gates_to_try:
            psi = random_state(n, 50 + trial)
            b = cuda(n, np.complex128)
            b.set_state(psi)
            b.apply(g)
            np.testing.assert_allclose(b.statevector(), full_matrix(g, n) @ psi, atol=1e-12)


def test_norm_and_inverse_on_gpu() -> None:
    from tests.tolerances import NORM_ATOL_COMPLEX64

    n = 12
    c = random_circuit(n, 200, 5)
    b = cuda(n)
    for g in c:
        b.apply(g)
    assert abs(b.amplitude_norm() - 1) < NORM_ATOL_COMPLEX64
    b.run(c.inverse())
    sv = b.statevector()
    assert abs(sv[0] - 1) < 1e-4


def _chi_square_ok(counts: dict[int, int], probs: np.ndarray, shots: int) -> bool:
    from scipy.stats import chisquare

    keep = probs > 1e-12
    observed = np.array([counts.get(i, 0) for i in np.flatnonzero(keep)], dtype=float)
    assert sum(counts.get(i, 0) for i in np.flatnonzero(~keep)) == 0, "sampled a zero-prob state"
    expected = probs[keep] / probs[keep].sum() * shots
    return bool(chisquare(observed, expected).pvalue > 1e-3)


@pytest.mark.parametrize("n", [3, 8, 12, 14])
@pytest.mark.parametrize("method", ["sample", "sample_cumsum"])
def test_sampling_matches_probabilities(n: int, method: str) -> None:
    c = random_circuit(n, 40, 300 + n)
    b = cuda(n)
    b.run(c)
    shots = 200_000
    counts = getattr(b, method)(shots, seed=11)
    assert sum(counts.values()) == shots
    assert _chi_square_ok(counts, b.probabilities(), shots)


def test_sampling_is_reproducible_with_seed() -> None:
    b = cuda(10)
    b.run(random_circuit(10, 30, 1))
    assert b.sample(1000, seed=3) == b.sample(1000, seed=3)


def test_sampling_bell_never_gives_odd_parity() -> None:
    b = cuda(2)
    b.run(Circuit(2).h(0).cx(0, 1))
    assert set(b.sample(10_000, seed=0)) == {0, 3}


@pytest.mark.parametrize(("dtype", "atol"), [(np.complex64, 1e-5), (np.complex128, 1e-10)])
def test_pauli_expectation_matches_oracle(dtype: type, atol: float) -> None:
    from tests.circuits import random_pauli

    for seed in range(30):
        n = 1 + seed % 10
        psi = random_state(n, seed)
        pauli = random_pauli(n, 1000 + seed)
        ref = NumpyBackend(n)
        ref.set_state(psi)
        b = cuda(n, dtype)
        b.set_state(psi)
        value = b.pauli_value(pauli)
        assert abs(value.imag) < atol, f"{pauli}: imaginary part {value.imag}"
        assert value.real == pytest.approx(ref.pauli_value(pauli).real, abs=atol)


def test_expectation_hamiltonian_and_norm() -> None:
    b = cuda(2)
    b.run(Circuit(2).h(0).cx(0, 1))
    terms = [(0.5, "ZZ"), (0.25, "XX"), (2.0, "II"), (1.0, "ZI"), (-0.5, "YY")]
    assert b.expectation_hamiltonian(terms) == pytest.approx(0.5 + 0.25 + 2.0 + 0.0 + 0.5,
                                                             abs=1e-6)
    assert b.amplitude_norm() == pytest.approx(1.0, abs=1e-6)


def test_host_copy_refused_above_limit() -> None:
    from qsim.backends.cuda_backend import CudaBackend

    b = CudaBackend(4)
    b.max_host_qubits = 3
    with pytest.raises(ValueError):
        b.statevector()
