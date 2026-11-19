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
