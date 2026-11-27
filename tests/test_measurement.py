"""Mid-circuit measurement, reset and classical conditions (W3), on both backends."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.stats import binomtest

from qsimgpu.algorithms import bitflip, teleport
from qsimgpu.backends import NumpyBackend
from qsimgpu.circuit import Circuit, Gate
from qsimgpu.execution import has_mid_circuit_measurement, run_shots
from qsimgpu.passes.fusion import fuse
from tests.circuits import random_circuit, random_state
from tests.tolerances import ATOL_COMPLEX64, ATOL_COMPLEX128

S2 = 1 / math.sqrt(2)


def _cuda(n: int) -> Any:
    from qsimgpu.backends.cuda_backend import CudaBackend

    return CudaBackend(n)


@pytest.fixture(params=["numpy", pytest.param("cuda", marks=pytest.mark.gpu)])
def make(request: pytest.FixtureRequest) -> Callable[[int], Any]:
    return NumpyBackend if request.param == "numpy" else _cuda


def _atol(b: Any) -> float:
    return ATOL_COMPLEX128 if isinstance(b, NumpyBackend) else ATOL_COMPLEX64


# --- the circuit IR ----------------------------------------------------------------

def test_measure_reset_and_conditions_are_validated() -> None:
    c = Circuit(2, n_clbits=2).h(0).measure(0, 1).reset(1).x(1).c_if([1], 1)
    assert c.gates[1] == Gate("measure", (0,), clbits=(1,))
    assert c.gates[3].condition == ((1,), 1)
    with pytest.raises(ValueError):
        Circuit(1, n_clbits=1).measure(0, 1)          # clbit out of range
    with pytest.raises(ValueError):
        Gate("measure", (0,))                         # needs a clbit
    with pytest.raises(ValueError):
        Gate("x", (0,), condition=((0,), 2))          # value doesn't fit in one bit
    with pytest.raises(ValueError):
        Circuit(1).c_if([0], 1)                       # no gate to condition


def test_inverse_refuses_measurement_and_conditions() -> None:
    for c in (Circuit(1, n_clbits=1).h(0).measure(0, 0), Circuit(1).reset(0),
              Circuit(1, n_clbits=1).x(0).c_if([0], 1)):
        with pytest.raises(ValueError):
            c.inverse()


def test_fusion_never_crosses_a_measurement() -> None:
    c = Circuit(2, n_clbits=1).h(0).cx(0, 1).measure(1, 0).h(1).x(0).c_if([0], 1).h(0)
    f = fuse(c, 2)
    names = [g.name for g in f]
    assert names[names.index("measure") + 1:].count("unitary") <= 1
    assert "measure" in names and any(g.condition for g in f)
    i = names.index("measure")
    assert all(set(g.qubits) for g in f.gates[:i])  # gates before stay before
    assert [g.name for g in f.gates[i:i + 1]] == ["measure"]


def test_mid_circuit_detection() -> None:
    assert not has_mid_circuit_measurement(Circuit(2, n_clbits=2).h(0).measure(0, 0).measure(1, 1))
    assert has_mid_circuit_measurement(Circuit(1, n_clbits=1).measure(0, 0).h(0))
    assert has_mid_circuit_measurement(Circuit(1).reset(0))
    assert has_mid_circuit_measurement(Circuit(1, n_clbits=1).x(0).c_if([0], 0))


# --- measurement on the backends ---------------------------------------------------------

def test_measuring_plus_is_fifty_fifty(make: Callable[[int], Any]) -> None:
    b = make(1)
    ones = 0
    for seed in range(2000):
        bits = b.run(Circuit(1, n_clbits=1).h(0).measure(0, 0), rng=seed)
        ones += bits[0]
        b.reset()
    # two-sided exact binomial test against p = 0.5
    assert binomtest(ones, 2000, 0.5).pvalue > 1e-3


def test_post_measurement_state_is_the_measured_basis_state(make: Callable[[int], Any]) -> None:
    for seed in range(20):
        n = 4
        b = make(n)
        b.set_state(random_state(n, seed))
        rng = np.random.default_rng(seed)
        outcome = b.measure_qubit(2, rng)
        sv = b.statevector()
        assert abs(np.linalg.norm(sv) - 1) < 1e-5
        idx = np.arange(1 << n)
        assert np.all(np.abs(sv[((idx >> 2) & 1) != outcome]) < 1e-7)


def test_reset_always_gives_zero(make: Callable[[int], Any]) -> None:
    for seed in range(20):
        b = make(3)
        b.set_state(random_state(3, seed))
        b.reset_qubit(1, np.random.default_rng(seed))
        p = b.probabilities()
        assert sum(p[i] for i in range(8) if (i >> 1) & 1) < 1e-10


def test_conditions_use_the_register_value(make: Callable[[int], Any]) -> None:
    c = Circuit(3, n_clbits=2).x(0).measure(0, 0).measure(1, 1)  # register = 0b01 = 1
    c.x(2).c_if([0, 1], 1).h(1).c_if([0, 1], 2)                  # first fires, second doesn't
    b = make(3)
    assert b.run(c, rng=0) == (1, 0)
    assert b.probabilities()[0b101] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(25))
def test_numpy_and_cuda_give_identical_bits_for_the_same_seed(seed: int) -> None:
    c = random_circuit(6, 40, seed=seed, allow_multi_controls=False)
    c = Circuit(6, list(c), n_clbits=5)
    for j, q in enumerate((0, 2, 4, 1, 3)):
        c.measure(q, j)
        c.h(q).ry(0.3 * (j + 1), (q + 1) % 6)
    assert NumpyBackend(6).run(c, rng=seed) == _cuda(6).run(c, rng=seed)


# --- shots ----------------------------------------------------------------------------

def test_run_shots_fast_path_matches_probabilities(make: Callable[[int], Any]) -> None:
    c = Circuit(2, n_clbits=2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    counts = run_shots(make(2), c, shots=4000, seed=1)
    assert set(counts) == {0b00, 0b11} and sum(counts.values()) == 4000


def test_run_shots_with_mid_circuit_measurement(make: Callable[[int], Any]) -> None:
    # measure, then flip qubit 1 if the result was 1: both clbits must always agree
    c = Circuit(2, n_clbits=2).h(0).measure(0, 0).x(1).c_if([0], 1).measure(1, 1)
    counts = run_shots(make(2), c, shots=500, seed=2)
    assert set(counts) <= {0b00, 0b11} and sum(counts.values()) == 500
    assert binomtest(counts.get(0b11, 0), 500, 0.5).pvalue > 1e-3


# --- algorithms ---------------------------------------------------------------------------

def test_teleportation_reproduces_random_states(make: Callable[[int], Any]) -> None:
    rng = np.random.default_rng(7)
    for trial in range(50):
        a, b2 = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psi = np.array([a, b2]) / np.linalg.norm([a, b2])
        backend = make(3)
        fidelity = teleport.teleport_and_check(backend, psi, rng=trial)
        assert fidelity > 1 - 10 * _atol(backend)


@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
def test_bitflip_code_corrects_any_single_flip(make: Callable[[int], Any],
                                               error_qubit: int | None) -> None:
    for seed in range(10):
        psi = random_state(1, seed)
        backend = make(5)
        recovered = bitflip.encode_flip_correct(backend, psi, error_qubit, rng=seed)
        overlap = abs(np.vdot(psi, recovered)) ** 2
        assert overlap > 1 - 10 * _atol(backend)
