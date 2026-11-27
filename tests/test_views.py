"""Step-by-step execution and state views (W2), on both backends."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qsimgpu import views
from qsimgpu.backends import NumpyBackend
from qsimgpu.circuit import Circuit, Gate
from tests.circuits import random_circuit, random_state
from tests.tolerances import ATOL_COMPLEX128, VIEW_ATOL_COMPLEX64

S2 = 1 / math.sqrt(2)


def _cuda(n: int) -> Any:
    from qsimgpu.backends.cuda_backend import CudaBackend

    return CudaBackend(n)


@pytest.fixture(params=["numpy", pytest.param("cuda", marks=pytest.mark.gpu)])
def make(request: pytest.FixtureRequest) -> Callable[[int], Any]:
    return NumpyBackend if request.param == "numpy" else _cuda


def _atol(backend: Any) -> float:
    return ATOL_COMPLEX128 if isinstance(backend, NumpyBackend) else VIEW_ATOL_COMPLEX64


# --- on_step -----------------------------------------------------------------------

def test_on_step_called_once_per_gate_in_order(make: Callable[[int], Any]) -> None:
    c = random_circuit(5, 30, seed=1)
    b = make(5)
    seen: list[tuple[int, Gate]] = []
    b.run(c, on_step=lambda i, g, backend: seen.append((i, g)))
    assert [i for i, _ in seen] == list(range(len(c)))
    assert [g for _, g in seen] == c.gates


def test_on_step_sees_the_state_after_each_gate(make: Callable[[int], Any]) -> None:
    c = Circuit(2).h(0).cx(0, 1)
    b = make(2)
    states: list[np.ndarray] = []
    b.run(c, on_step=lambda i, g, backend: states.append(backend.statevector()))
    np.testing.assert_allclose(states[0], [S2, S2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(states[1], [S2, 0, 0, S2], atol=1e-6)


def test_run_without_on_step_is_unchanged(make: Callable[[int], Any]) -> None:
    c = random_circuit(6, 40, seed=2)
    a, b = make(6), make(6)
    a.run(c)
    b.run(c, on_step=lambda *_: None)
    np.testing.assert_array_equal(a.statevector(), b.statevector())


# --- marginals ---------------------------------------------------------------------

def test_marginal_bit_order_follows_the_qubit_list(make: Callable[[int], Any]) -> None:
    b = make(3)
    b.run(Circuit(3).x(0))  # pattern 001: qubit 0 = 1
    np.testing.assert_allclose(views.marginal_probabilities(b, [0]), [0, 1], atol=1e-7)
    np.testing.assert_allclose(views.marginal_probabilities(b, [2, 0]), [0, 0, 1, 0], atol=1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_marginals_match_summing_the_full_distribution(make: Callable[[int], Any],
                                                       seed: int) -> None:
    n = 8
    rng = np.random.default_rng(seed)
    psi = random_state(n, seed)
    b = make(n)
    b.set_state(psi)
    k = int(rng.integers(1, 6))
    qubits = [int(q) for q in rng.permutation(n)[:k]]
    p = np.abs(psi) ** 2
    expected = np.zeros(1 << k)
    for i, pi in enumerate(p):
        expected[sum(((i >> q) & 1) << j for j, q in enumerate(qubits))] += pi
    np.testing.assert_allclose(views.marginal_probabilities(b, qubits), expected, atol=_atol(b))


def test_marginal_rejects_bad_qubit_lists() -> None:
    b = NumpyBackend(3)
    for bad in ([], [0, 0], [3], list(range(13))):
        with pytest.raises(ValueError):
            views.marginal_probabilities(b, bad)


# --- Bloch vectors, entanglement hint, snapshot ----------------------------------------

@pytest.mark.parametrize(("prep", "expected"), [
    (Circuit(1), (0, 0, 1)),                 # |0>
    (Circuit(1).x(0), (0, 0, -1)),           # |1>
    (Circuit(1).h(0), (1, 0, 0)),            # |+>
    (Circuit(1).h(0).s(0), (0, 1, 0)),       # |+i>
])
def test_bloch_vectors_of_known_states(make: Callable[[int], Any], prep: Circuit,
                                       expected: tuple[int, int, int]) -> None:
    b = make(1)
    b.run(prep)
    np.testing.assert_allclose(views.bloch_vector(b, 0), expected, atol=1e-6)
    assert views.entanglement_hint(b, 0) == pytest.approx(1.0, abs=1e-6)


def test_half_of_a_bell_pair_has_bloch_length_zero(make: Callable[[int], Any]) -> None:
    b = make(2)
    b.run(Circuit(2).h(0).cx(0, 1))
    for q in (0, 1):
        np.testing.assert_allclose(views.bloch_vector(b, q), (0, 0, 0), atol=1e-6)
        assert views.entanglement_hint(b, q) == pytest.approx(0.0, abs=1e-6)


def test_snapshot_refuses_large_states() -> None:
    b = NumpyBackend(13)
    with pytest.raises(ValueError, match="snapshot"):
        views.snapshot(b)
    assert views.snapshot(b, max_qubits=13).shape == (1 << 13,)


# --- step trace --------------------------------------------------------------------------

def test_trace_has_one_entry_per_gate_plus_the_start(make: Callable[[int], Any]) -> None:
    c = random_circuit(10, 20, seed=5)
    tr = views.step_trace(make(10), c)
    assert tr["trace_version"] == 1 and tr["n_qubits"] == 10
    assert len(tr["steps"]) == len(c) + 1
    assert tr["steps"][0]["index"] == -1 and tr["steps"][0]["gate"] is None
    assert [s["index"] for s in tr["steps"][1:]] == list(range(len(c)))
    assert json.loads(json.dumps(tr)) == tr  # plain JSON, round-trips exactly


def test_trace_uses_marginals_above_the_full_probability_limit() -> None:
    c = random_circuit(6, 10, seed=3)
    tr = views.step_trace(NumpyBackend(6), c, full_probabilities_up_to=4, marginal_qubits=[0, 5])
    step = tr["steps"][-1]
    assert "probabilities" not in step
    assert step["marginals"]["qubits"] == [0, 5] and len(step["marginals"]["probabilities"]) == 4


def test_trace_serialises_unitary_gates() -> None:
    c = Circuit(2).unitary(np.array([[0, 1j], [1j, 0]]), [1])
    gate = views.step_trace(NumpyBackend(2), c)["steps"][1]["gate"]
    assert gate["name"] == "unitary" and gate["matrix"] == [[[0.0, 0.0], [0.0, 1.0]],
                                                             [[0.0, 1.0], [0.0, 0.0]]]


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(4))
def test_numpy_and_cuda_traces_agree(seed: int) -> None:
    c = random_circuit(10, 20, seed=100 + seed)
    a = views.step_trace(NumpyBackend(10), c)
    b = views.step_trace(_cuda(10), c)
    assert views.trace_max_difference(a, b) < VIEW_ATOL_COMPLEX64
