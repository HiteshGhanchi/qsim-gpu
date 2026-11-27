"""Views of a simulator's state that a visual front end can show, plus step-by-step traces.

All functions take any Backend (NumPy or CUDA) and never copy a large state to the host:
- marginal_probabilities: distribution of a few chosen qubits (reduced on the GPU for CUDA).
- bloch_vector: (<X>, <Y>, <Z>) of one qubit, from three Pauli expectations.
- entanglement_hint: length of that Bloch vector.
- snapshot: the full statevector, only for small states.
- step_trace: run a circuit and record these views after every gate, as plain JSON data.

Trace format (trace_version 1):
    {"trace_version": 1, "n_qubits": n, "steps": [step, ...]}
    step = {"index": -1 for the starting state, else the gate's position,
            "gate": null or {"name", "targets", "controls", "params"[, "matrix"]},
            "probabilities": [...]                      (when n <= full_probabilities_up_to)
            "marginals": {"qubits": [...], "probabilities": [...]}   (otherwise, if requested)
            "bloch": [[x, y, z] for each qubit]          (when bloch=True)}
Matrices are written as nested [real, imag] pairs. Cost per step: one full-state pass per
marginal, plus three passes per qubit for Bloch vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qsimgpu.backends.base import Backend
from qsimgpu.circuit import Circuit, Gate

TRACE_VERSION = 1


def _pauli_on(backend: Backend, q: int, ch: str) -> str:
    n = backend.n_qubits
    if not 0 <= q < n:
        raise ValueError(f"qubit {q} out of range for {n} qubits")
    chars = ["I"] * n
    chars[n - 1 - q] = ch  # leftmost character = highest qubit
    return "".join(chars)


def marginal_probabilities(backend: Backend, qubits: Sequence[int]) -> NDArray[np.float64]:
    """Probabilities of the 2^k outcomes of `qubits`; bit j of the index belongs to qubits[j]."""
    return np.asarray(backend.marginal_probabilities(qubits), dtype=np.float64)


def bloch_vector(backend: Backend, q: int) -> tuple[float, float, float]:
    """(<X>, <Y>, <Z>) of qubit q. Length 1 for a qubit in a pure state of its own."""
    x, y, z = (backend.expectation_pauli(_pauli_on(backend, q, ch)) for ch in "XYZ")
    return (x, y, z)


def entanglement_hint(backend: Backend, q: int) -> float:
    """Length of qubit q's Bloch vector, between 0 and 1.

    While the whole register is in a pure state (always true in this simulator, which has no
    noise), a length below 1 means qubit q is entangled with the other qubits, and 0 means it
    is maximally entangled (half of a Bell pair). It does not say *which* qubits it is
    entangled with, and it would not mean the same thing for a mixed (noisy) state.
    """
    return math.sqrt(sum(c * c for c in bloch_vector(backend, q)))


def snapshot(backend: Backend, max_qubits: int = 12) -> NDArray[np.complexfloating[Any, Any]]:
    """Copy of the full statevector, refused above max_qubits (2^n values would be copied)."""
    if backend.n_qubits > max_qubits:
        raise ValueError(f"snapshot refused: {backend.n_qubits} qubits is above max_qubits="
                         f"{max_qubits}; use marginal_probabilities or bloch_vector instead")
    return backend.statevector()


def gate_to_dict(gate: Gate) -> dict[str, Any]:
    d: dict[str, Any] = {"name": gate.name, "targets": list(gate.targets),
                         "controls": list(gate.controls), "params": list(gate.params)}
    if gate.matrix is not None:
        d["matrix"] = [[[float(z.real), float(z.imag)] for z in row] for row in gate.matrix]
    return d


def _views(backend: Backend, full_up_to: int, marginal_qubits: Sequence[int] | None,
           bloch: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = backend.n_qubits
    if n <= full_up_to:
        out["probabilities"] = [float(p) for p in backend.probabilities()]
    elif marginal_qubits:
        out["marginals"] = {"qubits": list(marginal_qubits),
                            "probabilities": [float(p) for p in
                                              marginal_probabilities(backend, marginal_qubits)]}
    if bloch:
        out["bloch"] = [list(bloch_vector(backend, q)) for q in range(n)]
    return out


def step_trace(backend: Backend, circuit: Circuit, *, full_probabilities_up_to: int = 12,
               marginal_qubits: Sequence[int] | None = None, bloch: bool = True) -> dict[str, Any]:
    """Reset the backend, run `circuit` one gate at a time and record the views after each gate.

    Full probabilities are recorded while n <= full_probabilities_up_to; above that, only the
    marginals of `marginal_qubits` (if given). The result is plain JSON-serialisable data.
    """
    backend.reset()
    steps = [{"index": -1, "gate": None,
              **_views(backend, full_probabilities_up_to, marginal_qubits, bloch)}]

    def record(i: int, gate: Gate, b: Backend) -> None:
        steps.append({"index": i, "gate": gate_to_dict(gate),
                      **_views(b, full_probabilities_up_to, marginal_qubits, bloch)})

    backend.run(circuit, on_step=record)
    return {"trace_version": TRACE_VERSION, "n_qubits": backend.n_qubits, "steps": steps}


def trace_max_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Largest absolute difference between the numbers of two traces of the same circuit."""
    if (a["trace_version"], a["n_qubits"], len(a["steps"])) != (
            b["trace_version"], b["n_qubits"], len(b["steps"])):
        raise ValueError("traces describe different runs")
    worst = 0.0
    for sa, sb in zip(a["steps"], b["steps"], strict=True):
        if sa["gate"] != sb["gate"]:
            raise ValueError(f"step {sa['index']}: different gates")
        for key in ("probabilities", "bloch"):
            if key in sa:
                diff = np.abs(np.asarray(sa[key]) - np.asarray(sb[key]))
                worst = max(worst, float(diff.max()))
        if "marginals" in sa:
            diff = np.abs(np.asarray(sa["marginals"]["probabilities"])
                          - np.asarray(sb["marginals"]["probabilities"]))
            worst = max(worst, float(diff.max()))
    return worst
