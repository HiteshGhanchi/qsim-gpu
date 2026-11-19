"""Fusion must not change what a circuit does, only how many passes it takes."""

from __future__ import annotations

import numpy as np
import pytest

from qsim.algorithms.qft import qft
from qsim.backends import NumpyBackend
from qsim.circuit import Circuit
from qsim.passes.fusion import block_matrix, fuse
from tests.circuits import random_circuit


def final_state(c: Circuit) -> np.ndarray:
    b = NumpyBackend(c.n_qubits)
    b.run(c)
    return b.statevector()


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(6))
def test_fused_equals_unfused_on_oracle(m: int, seed: int) -> None:
    n = 3 + seed
    c = random_circuit(n, 120, 7000 + seed)
    f = fuse(c, m)
    assert len(f) <= len(c)
    np.testing.assert_allclose(final_state(f), final_state(c), atol=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_fusing_qft_reduces_passes(m: int) -> None:
    c = qft(10)
    f = fuse(c, m)
    assert len(f) < len(c)
    np.testing.assert_allclose(final_state(f), final_state(c), atol=1e-12)


def test_wide_gates_pass_through_unchanged() -> None:
    c = Circuit(5).h(0).mcz([0, 1, 2, 3], 4).h(0)
    f = fuse(c, 2)
    assert [g.name for g in f] == ["h", "z", "h"]


def test_single_gate_blocks_are_not_wrapped() -> None:
    c = Circuit(4).cz(0, 1).cz(2, 3)  # second gate would make 4 qubits > m=2
    f = fuse(c, 2)
    assert [g.name for g in f] == ["cz", "cz"]


def test_block_matrix_of_cx() -> None:
    from qsim.circuit import Gate

    m = block_matrix([Gate("cx", (1,), (0,))], (0, 1))
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    np.testing.assert_allclose(m, expected)


def test_rejects_too_wide_blocks() -> None:
    with pytest.raises(ValueError):
        fuse(Circuit(2), 6)


@pytest.mark.gpu
@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_fused_circuit_on_gpu_matches_oracle(m: int) -> None:
    from qsim.backends.cuda_backend import CudaBackend

    c = random_circuit(14, 150, 31 + m)
    b = CudaBackend(14)
    b.run(fuse(c, m))
    np.testing.assert_allclose(b.statevector(), final_state(c), atol=1e-5)
