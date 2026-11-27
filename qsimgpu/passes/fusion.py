"""Gate fusion: merge runs of consecutive gates that together touch at most m qubits.

Every gate is one full pass over the state in memory. If gates 1..k only involve qubits
{a, b, c}, their product is a single 8x8 matrix, and applying that matrix is still one pass.
Fusion trades a little arithmetic (bigger matrices) for fewer passes over memory, which is
the right trade when the simulator is memory-bound.

The pass is greedy and keeps gate order: it grows a block while the union of qubits stays
within max_qubits, and closes the block when the next gate would exceed it. A block of one
gate is emitted unchanged, so the backend can still use its specialised kernel for it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qsimgpu.circuit import Circuit, Gate


def block_matrix(block: list[Gate], qubits: tuple[int, ...]) -> NDArray[np.complex128]:
    """Matrix of a list of gates restricted to `qubits` (qubits[0] = least significant bit)."""
    from qsimgpu.backends.numpy_backend import NumpyBackend

    k = len(qubits)
    local = {q: i for i, q in enumerate(qubits)}
    remapped = [Gate(g.name, tuple(local[q] for q in g.targets),
                     tuple(local[q] for q in g.controls), g.params, g.matrix) for g in block]
    out = np.zeros((1 << k, 1 << k), dtype=np.complex128)
    sim = NumpyBackend(k)
    for col in range(1 << k):  # column j = the block applied to basis state |j>
        sim.reset(col)
        for g in remapped:
            sim.apply(g)
        out[:, col] = sim.statevector()
    return out


def fuse(circuit: Circuit, max_qubits: int = 2) -> Circuit:
    """Return an equivalent circuit where consecutive gates are merged into unitary gates."""
    if not 1 <= max_qubits <= 5:
        raise ValueError("max_qubits must be between 1 and 5 (largest kernel matrix is 32x32)")
    out = Circuit(circuit.n_qubits)
    block: list[Gate] = []
    block_qubits: set[int] = set()

    def flush() -> None:
        if len(block) == 1:
            out.append(block[0])
        elif block:
            qs = tuple(sorted(block_qubits))
            out.append(Gate("unitary", qs, matrix=block_matrix(block, qs)))
        block.clear()
        block_qubits.clear()

    for g in circuit:
        qs = set(g.qubits)
        if len(qs) > max_qubits:  # too wide to fuse at all: emit as-is
            flush()
            out.append(g)
            continue
        if len(block_qubits | qs) > max_qubits:
            flush()
        block.append(g)
        block_qubits |= qs
    flush()
    return out
