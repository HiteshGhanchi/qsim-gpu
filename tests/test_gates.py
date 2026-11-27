"""Gate matrices: unitary, match the documented definitions, inverses work."""

import math

import numpy as np
import pytest

from qsimgpu import gates
from qsimgpu.circuit import Gate

S2 = 1 / math.sqrt(2)


@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "swap"])
def test_fixed_gates_are_unitary(name: str) -> None:
    assert gates.is_unitary(gates.base_matrix(name))


@pytest.mark.parametrize("name", ["rx", "ry", "rz", "p"])
@pytest.mark.parametrize("theta", [0.0, 0.3, -1.2, math.pi, 2 * math.pi])
def test_rotation_gates_are_unitary(name: str, theta: float) -> None:
    assert gates.is_unitary(gates.base_matrix(name, (theta,)))


def test_definitions_match_qiskit_conventions() -> None:
    np.testing.assert_allclose(gates.base_matrix("h"), [[S2, S2], [S2, -S2]])
    np.testing.assert_allclose(gates.base_matrix("y"), [[0, -1j], [1j, 0]])
    np.testing.assert_allclose(gates.base_matrix("s"), [[1, 0], [0, 1j]])
    np.testing.assert_allclose(gates.base_matrix("t"), [[1, 0], [0, np.exp(1j * math.pi / 4)]])
    t = 0.7
    np.testing.assert_allclose(gates.base_matrix("rz", (t,)),
                               np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)]))
    np.testing.assert_allclose(gates.base_matrix("ry", (math.pi,)), [[0, -1], [1, 0]], atol=1e-15)
    np.testing.assert_allclose(gates.base_matrix("rx", (math.pi,)), [[0, -1j], [-1j, 0]],
                               atol=1e-15)


def test_rz_equals_p_up_to_global_phase() -> None:
    t = 1.1
    np.testing.assert_allclose(gates.base_matrix("rz", (t,)),
                               np.exp(-0.5j * t) * gates.base_matrix("p", (t,)))


def test_known_identities() -> None:
    h, x, z = (gates.base_matrix(n) for n in "hxz")
    s, t = gates.base_matrix("s"), gates.base_matrix("t")
    np.testing.assert_allclose(h @ h, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(h @ z @ h, x, atol=1e-15)
    np.testing.assert_allclose(s @ s, z, atol=1e-15)
    np.testing.assert_allclose(t @ t, s, atol=1e-15)


@pytest.mark.parametrize(("name", "params"), [
    ("h", ()), ("s", ()), ("sdg", ()), ("t", ()), ("tdg", ()), ("rx", (0.4,)),
    ("ry", (-0.9,)), ("rz", (2.0,)), ("p", (1.3,)),
])
def test_inverse_matrix(name: str, params: tuple[float, ...]) -> None:
    g = Gate(name, (0,), params=params)
    np.testing.assert_allclose(g.inverse().target_matrix() @ g.target_matrix(), np.eye(2),
                               atol=1e-14)


def test_gate_kinds() -> None:
    assert gates.SPECS["cz"].kind == "diagonal"
    assert gates.SPECS["ccx"].kind == "permutation"
    assert gates.SPECS["h"].kind == "general"
