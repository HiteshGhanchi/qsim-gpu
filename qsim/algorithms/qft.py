"""Quantum Fourier Transform.

QFT|x> = 1/sqrt(N) * sum_k exp(2*pi*i*x*k/N) |k>,  N = 2^n.

Circuit: for each qubit from the top down, an H followed by controlled-phase rotations from
the lower qubits, then swaps to reverse the qubit order. Interference turns a periodic
pattern of amplitudes into sharp peaks, which is what Shor's algorithm reads out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from qsim.circuit import Circuit, Gate


def qft_gates(qubits: Sequence[int], inverse: bool = False, swaps: bool = True) -> list[Gate]:
    """QFT on `qubits` (qubits[0] = least significant bit of x) as a list of gates."""
    qs = list(qubits)
    n = len(qs)
    gate_list: list[Gate] = []
    for j in reversed(range(n)):
        gate_list.append(Gate("h", (qs[j],)))
        for k in reversed(range(j)):
            gate_list.append(Gate("cp", (qs[j],), (qs[k],), (math.pi / 2 ** (j - k),)))
    if swaps:
        for i in range(n // 2):
            gate_list.append(Gate("swap", (qs[i], qs[n - 1 - i])))
    if inverse:
        gate_list = [g.inverse() for g in reversed(gate_list)]
    return gate_list


def qft(n_qubits: int, inverse: bool = False) -> Circuit:
    """A circuit that applies the QFT (or its inverse) to all n qubits."""
    return Circuit(n_qubits, qft_gates(range(n_qubits), inverse=inverse))
