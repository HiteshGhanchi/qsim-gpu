"""Random circuit generator shared by the tests and benchmarks."""

from __future__ import annotations

import math

import numpy as np

from qsim.circuit import Circuit

ONE_QUBIT = ["h", "x", "y", "z", "s", "sdg", "t", "tdg"]
ONE_QUBIT_PARAM = ["rx", "ry", "rz", "p"]
TWO_QUBIT = ["cx", "cz", "cp", "swap"]


def random_circuit(n_qubits: int, depth: int, seed: int, *, allow_multi_controls: bool = True,
                   only_one_qubit: bool = False) -> Circuit:
    """A random circuit of `depth` gates drawn from the whole gate set.

    With allow_multi_controls, some gates get extra control qubits (generic controlled gates).
    """
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits)
    pool = ONE_QUBIT + ONE_QUBIT_PARAM
    if not only_one_qubit and n_qubits >= 2:
        pool = pool + TWO_QUBIT
    if not only_one_qubit and n_qubits >= 3:
        pool = pool + ["ccx"]
    for _ in range(depth):
        name = str(rng.choice(pool))
        qs = [int(q) for q in rng.permutation(n_qubits)]
        angle = float(rng.uniform(-math.pi, math.pi))
        if name in ONE_QUBIT or name in ONE_QUBIT_PARAM:
            controls: list[int] = []
            if allow_multi_controls and not only_one_qubit and n_qubits >= 2 and rng.random() < 0.2:
                controls = qs[1 : 1 + int(rng.integers(1, min(3, n_qubits - 1) + 1))]
            params = [angle] if name in ONE_QUBIT_PARAM else []
            c.add(name, [qs[0]], controls, params)
        elif name == "swap":
            c.swap(qs[0], qs[1])
        elif name == "cp":
            c.cp(angle, qs[0], qs[1])
        elif name in ("cx", "cz"):
            c.add(name, [qs[1]], [qs[0]])
        elif name == "ccx":
            c.ccx(qs[0], qs[1], qs[2])
    return c


def random_state(n_qubits: int, seed: int) -> np.ndarray:
    """A random normalised complex128 statevector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1 << n_qubits) + 1j * rng.standard_normal(1 << n_qubits)
    return v / np.linalg.norm(v)


def random_pauli(n_qubits: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join(str(ch) for ch in rng.choice(list("IXYZ"), size=n_qubits))
