"""Gate definitions: exact matrices and properties for every supported gate.

All matrices follow Qiskit's conventions and are complex128 (the backends cast as needed).

Multi-target matrices are little-endian over the target list: bit i of a row/column index
belongs to targets[i]. So for a 2-target gate on (a, b), row index 0b10 means
"qubit b = 1, qubit a = 0".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.complex128]
Kind = Literal["diagonal", "permutation", "general", "nonunitary"]


@dataclass(frozen=True)
class GateSpec:
    """Static facts about one gate name.

    n_targets:  number of target qubits (None = any, used by "unitary").
    n_controls: exact number of controls, or None = any number (generic controlled gate).
    n_params:   number of angle parameters.
    base:       the gate that acts on the targets when all controls are 1 (cx -> "x").
    kind:       memory-access family, which decides the specialised GPU kernel.
    """

    n_targets: int | None
    n_controls: int | None
    n_params: int
    base: str
    kind: Kind


SPECS: dict[str, GateSpec] = {
    "h": GateSpec(1, None, 0, "h", "general"),
    "x": GateSpec(1, None, 0, "x", "permutation"),
    "y": GateSpec(1, None, 0, "y", "general"),
    "z": GateSpec(1, None, 0, "z", "diagonal"),
    "s": GateSpec(1, None, 0, "s", "diagonal"),
    "sdg": GateSpec(1, None, 0, "sdg", "diagonal"),
    "t": GateSpec(1, None, 0, "t", "diagonal"),
    "tdg": GateSpec(1, None, 0, "tdg", "diagonal"),
    "rx": GateSpec(1, None, 1, "rx", "general"),
    "ry": GateSpec(1, None, 1, "ry", "general"),
    "rz": GateSpec(1, None, 1, "rz", "diagonal"),
    "p": GateSpec(1, None, 1, "p", "diagonal"),
    "cx": GateSpec(1, 1, 0, "x", "permutation"),
    "cz": GateSpec(1, 1, 0, "z", "diagonal"),
    "cp": GateSpec(1, 1, 1, "p", "diagonal"),
    "ccx": GateSpec(1, 2, 0, "x", "permutation"),
    "swap": GateSpec(2, None, 0, "swap", "permutation"),
    "unitary": GateSpec(None, 0, 0, "unitary", "general"),
    # Non-unitary operations: handled by Backend.run, never by a matrix kernel.
    "measure": GateSpec(1, 0, 0, "measure", "nonunitary"),
    "reset": GateSpec(1, 0, 0, "reset", "nonunitary"),
}
NONUNITARY = frozenset({"measure", "reset"})

_SQ2 = 1.0 / math.sqrt(2.0)


def _m(rows: list[list[complex]]) -> Matrix:
    return np.array(rows, dtype=np.complex128)


def _phase(phi: float) -> Matrix:
    return _m([[1, 0], [0, np.exp(1j * phi)]])


_FIXED: dict[str, Matrix] = {
    "h": _m([[_SQ2, _SQ2], [_SQ2, -_SQ2]]),
    "x": _m([[0, 1], [1, 0]]),
    "y": _m([[0, -1j], [1j, 0]]),
    "z": _m([[1, 0], [0, -1]]),
    "s": _phase(math.pi / 2),
    "sdg": _phase(-math.pi / 2),
    "t": _phase(math.pi / 4),
    "tdg": _phase(-math.pi / 4),
    "swap": _m([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
}


def base_matrix(base: str, params: tuple[float, ...] = ()) -> Matrix:
    """Matrix of a base gate (no controls). `base` is a GateSpec.base name, not "unitary"."""
    if base in _FIXED:
        return _FIXED[base].copy()
    (theta,) = params
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if base == "rx":
        return _m([[c, -1j * s], [-1j * s, c]])
    if base == "ry":
        return _m([[c, -s], [s, c]])
    if base == "rz":
        return _m([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])
    if base == "p":
        return _phase(theta)
    raise ValueError(f"no matrix for base gate {base!r}")


_SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "ccx", "swap"}
_SWAP_INVERSE = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
_NEGATE_ANGLE = {"rx", "ry", "rz", "p", "cp"}


def inverse_name_params(name: str, params: tuple[float, ...]) -> tuple[str, tuple[float, ...]]:
    """(name, params) of the inverse gate. Not valid for "unitary" (use the conjugate transpose)."""
    if name in _SELF_INVERSE:
        return name, params
    if name in _SWAP_INVERSE:
        return _SWAP_INVERSE[name], params
    if name in _NEGATE_ANGLE:
        return name, tuple(-p for p in params)
    raise ValueError(f"no inverse rule for {name!r}")


def is_unitary(m: Matrix, atol: float = 1e-10) -> bool:
    """True if m is square and m^dagger m = I within atol."""
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=atol))
