"""Known-answer tests for GHZ, H-H, Grover, QFT and Shor-15, on both backends."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from qsim.algorithms import grover as gr
from qsim.algorithms import shor15
from qsim.algorithms.qft import qft
from qsim.backends import NumpyBackend
from qsim.circuit import Circuit

BackendFactory = Callable[[int], object]


def _cuda(n: int):  # type: ignore[no-untyped-def]
    from qsim.backends.cuda_backend import CudaBackend

    return CudaBackend(n)


@pytest.fixture(params=["numpy", pytest.param("cuda", marks=pytest.mark.gpu)])
def make(request: pytest.FixtureRequest) -> BackendFactory:
    return NumpyBackend if request.param == "numpy" else _cuda


def run(make: BackendFactory, c: Circuit):  # type: ignore[no-untyped-def]
    b = make(c.n_qubits)
    b.run(c)  # type: ignore[attr-defined]
    return b


# --- GHZ and H-H ---------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 5, 12, 24])
def test_ghz_only_two_nonzero_amplitudes(make: BackendFactory, n: int) -> None:
    if n > 20 and make is NumpyBackend:
        pytest.skip("oracle too slow/large at 24 qubits on CPU")
    c = Circuit(n).h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    b = run(make, c)
    counts = b.sample(5000, seed=1)
    assert set(counts) == {0, (1 << n) - 1}
    if n <= 20:
        p = b.probabilities()
        assert p[0] == pytest.approx(0.5, abs=1e-6) and p[-1] == pytest.approx(0.5, abs=1e-6)


def test_h_h_returns_zero(make: BackendFactory) -> None:
    p = run(make, Circuit(1).h(0).h(0)).probabilities()
    np.testing.assert_allclose(p, [1, 0], atol=1e-6)


# --- Grover --------------------------------------------------------------------

def test_grover_two_qubits_is_certain_after_one_round(make: BackendFactory) -> None:
    for marked in range(4):
        p = run(make, gr.grover(2, marked, iterations=1)).probabilities()
        assert p[marked] == pytest.approx(1.0, abs=1e-5)


def test_grover_three_qubits_two_rounds(make: BackendFactory) -> None:
    p = run(make, gr.grover(3, 5, iterations=2)).probabilities()
    assert p[5] == pytest.approx(0.9453, abs=1e-3)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_grover_matches_theory(make: BackendFactory, n: int, k: int) -> None:
    marked = (0b1011011011 >> (10 - n)) & ((1 << n) - 1)
    p = run(make, gr.grover(n, marked, iterations=k)).probabilities()
    assert p[marked] == pytest.approx(gr.success_probability(n, k), abs=1e-5)


def test_grover_optimal_iterations_are_near_certain() -> None:
    for n in range(2, 21):
        k = gr.optimal_iterations(n)
        assert gr.success_probability(n, k) > 0.9
        x = math.pi / 4 * math.sqrt(2**n)
        assert k in {math.floor(x), round(x)}


# --- QFT -----------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_qft_of_basis_states_matches_formula(make: BackendFactory, n: int) -> None:
    N = 2**n
    k = np.arange(N)
    xs = range(N) if n <= 5 else [0, 1, 37, 128, 255]
    for x in xs:
        b = make(n)
        b.reset(x)
        b.run(qft(n))
        expected = np.exp(2j * np.pi * x * k / N) / np.sqrt(N)
        np.testing.assert_allclose(b.statevector(), expected, atol=1e-5)


@pytest.mark.parametrize("r", [2, 4, 8, 16])
def test_qft_of_a_comb_peaks_at_multiples_of_n_over_r(r: int) -> None:
    n = 8
    N = 2**n
    comb = np.zeros(N, dtype=complex)
    comb[3 % r::r] = 1  # every r-th entry, starting at offset 3 mod r
    comb /= np.linalg.norm(comb)
    b = NumpyBackend(n)
    b.set_state(comb)
    b.run(qft(n))
    p = b.probabilities()
    peaks = np.flatnonzero(p > 1e-9)
    assert peaks.tolist() == list(range(0, N, N // r))


def test_inverse_qft_undoes_qft(make: BackendFactory) -> None:
    c = Circuit(6).h(0).t(0).cx(0, 3).ry(0.4, 5)
    c.extend(qft(6)).extend(qft(6, inverse=True)).extend(c.inverse())
    p = run(make, c).probabilities()
    assert p[0] == pytest.approx(1.0, abs=1e-5)


# --- Shor, N = 15 ----------------------------------------------------------------

@pytest.mark.parametrize("a", shor15.VALID_A)
def test_modular_multiplication_is_correct(a: int) -> None:
    for power in range(1, 5):
        for y in range(1, 15):
            c = Circuit(5)
            c.x(0)  # control qubit 0 = 1
            for i in range(4):
                if (y >> i) & 1:
                    c.x(1 + i)
            shor15.add_controlled_modmul(c, a, power, 0, [1, 2, 3, 4])
            b = NumpyBackend(5)
            b.run(c)
            out = int(np.argmax(b.probabilities())) >> 1
            assert out == (pow(a, power) * y) % 15


def test_modular_multiplication_does_nothing_when_control_is_zero() -> None:
    c = Circuit(5).x(2)
    shor15.add_controlled_modmul(c, 7, 1, 0, [1, 2, 3, 4])
    b = NumpyBackend(5)
    b.run(c)
    assert int(np.argmax(b.probabilities())) == 0b00100


def test_shor_a7_peaks_at_multiples_of_64(make: BackendFactory) -> None:
    p = shor15.input_distribution(run(make, shor15.shor15_circuit(7)).probabilities())
    np.testing.assert_allclose(p[[0, 64, 128, 192]], 0.25, atol=1e-5)
    assert p.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(("a", "r"), [(7, 4), (2, 4), (11, 2), (4, 2), (8, 4), (13, 4)])
def test_shor_recovers_period_and_factors(make: BackendFactory, a: int, r: int) -> None:
    result = shor15.run_shor15(make(12), a, shots=512, seed=a)  # type: ignore[arg-type]
    assert result.period == r
    if r == 4 or a == 11:
        assert result.factors == (3, 5)
    else:  # a = 4: 4^(2/2) = 4, gcd(3, 15) = 3 and gcd(5, 15) = 5 -> still works
        assert result.factors in {(3, 5), None}


def test_post_processing_examples() -> None:
    assert shor15.period_from_measurement(64, 7) == 4
    assert shor15.period_from_measurement(128, 7) == 4  # 1/2 -> r=2 fails, 4 works
    assert shor15.period_from_measurement(128, 11) == 2
    assert shor15.period_from_measurement(0, 7) is None
    assert shor15.factors_from_period(7, 4) == (3, 5)
