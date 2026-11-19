"""Circuit IR: validation, builders, inverse, depth, op counts, Gate equality."""

import numpy as np
import pytest

from qsim.circuit import Circuit, Gate


def test_builders_use_canonical_encoding() -> None:
    c = Circuit(3).h(0).cx(0, 1).cp(0.5, 1, 2).ccx(0, 1, 2).swap(0, 2)
    assert c.gates[1] == Gate("cx", (1,), (0,))
    assert c.gates[2] == Gate("cp", (2,), (1,), (0.5,))
    assert c.gates[3] == Gate("ccx", (2,), (0, 1))
    assert c.gates[4] == Gate("swap", (0, 2))


@pytest.mark.parametrize("bad", [
    lambda: Gate("nope", (0,)),
    lambda: Gate("h", (0, 1)),
    lambda: Gate("cx", (1,)),             # missing control
    lambda: Gate("rx", (0,)),             # missing angle
    lambda: Gate("cx", (1,), (1,)),       # control == target
    lambda: Gate("h", (-1,)),
    lambda: Gate("unitary", (0,)),        # missing matrix
    lambda: Gate("unitary", (0,), matrix=np.ones((2, 2))),  # not unitary
    lambda: Gate("h", (0,), matrix=np.eye(2)),
])
def test_invalid_gates_rejected(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        bad()


def test_out_of_range_qubit_rejected() -> None:
    with pytest.raises(ValueError):
        Circuit(2).h(2)


def test_gate_equality_and_hash_with_matrix() -> None:
    m = np.eye(4)
    a = Gate("unitary", (0, 1), matrix=m)
    b = Gate("unitary", (0, 1), matrix=m.copy())
    assert a == b and hash(a) == hash(b)
    assert a != Gate("unitary", (0, 1), matrix=m[[1, 0, 2, 3]])
    assert len({a, b}) == 1


def test_inverse_reverses_and_inverts() -> None:
    c = Circuit(2).h(0).s(1).rx(0.3, 0).cx(0, 1)
    inv = c.inverse()
    assert [g.name for g in inv] == ["cx", "rx", "sdg", "h"]
    assert inv.gates[1].params == (-0.3,)


def test_depth_and_count_ops() -> None:
    c = Circuit(3).h(0).h(1).h(2).cx(0, 1).cx(1, 2)
    assert c.depth() == 3
    assert c.count_ops() == {"h": 3, "cx": 2}
    assert Circuit(2).depth() == 0
