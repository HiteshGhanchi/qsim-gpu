"""Grover search for one marked item among N = 2^n.

Start in the equal superposition, then repeat k times:
  oracle:    flip the sign of the marked pattern (multi-controlled Z, wrapped in X gates
             on the qubits where the marked pattern has a 0),
  diffusion: reflect every amplitude about the mean (H, X, multi-controlled Z, X, H).
Each round moves amplitude from the unmarked patterns to the marked one. After k rounds the
success probability is sin^2((2k+1) * theta) with theta = asin(1/sqrt(N)), so about
(pi/4)*sqrt(N) rounds are needed: a square-root speedup, not an exponential one.
"""

from __future__ import annotations

import math

from qsimgpu.circuit import Circuit


def _flip_all_ones(c: Circuit) -> None:
    """Sign flip on |1...1>: Z with every other qubit as a control."""
    n = c.n_qubits
    if n == 1:
        c.z(0)
    else:
        c.mcz(list(range(n - 1)), n - 1)


def add_oracle(c: Circuit, marked: int) -> None:
    zeros = [q for q in range(c.n_qubits) if not (marked >> q) & 1]
    for q in zeros:
        c.x(q)
    _flip_all_ones(c)
    for q in zeros:
        c.x(q)


def add_diffusion(c: Circuit) -> None:
    n = c.n_qubits
    for q in range(n):
        c.h(q)
        c.x(q)
    _flip_all_ones(c)
    for q in range(n):
        c.x(q)
        c.h(q)


def success_probability(n_qubits: int, iterations: int) -> float:
    """Theoretical chance of measuring the marked item after `iterations` rounds."""
    theta = math.asin(1 / math.sqrt(2**n_qubits))
    return math.sin((2 * iterations + 1) * theta) ** 2


def optimal_iterations(n_qubits: int) -> int:
    """Pick floor or round of (pi/4)*sqrt(N), whichever gives the higher success probability."""
    x = math.pi / 4 * math.sqrt(2**n_qubits)
    candidates = {max(1, math.floor(x)), max(1, round(x))}
    return max(sorted(candidates), key=lambda k: success_probability(n_qubits, k))


def grover(n_qubits: int, marked: int, iterations: int | None = None) -> Circuit:
    if not 0 <= marked < 2**n_qubits:
        raise ValueError("marked item out of range")
    k = optimal_iterations(n_qubits) if iterations is None else iterations
    c = Circuit(n_qubits)
    for q in range(n_qubits):
        c.h(q)
    for _ in range(k):
        add_oracle(c, marked)
        add_diffusion(c)
    return c
