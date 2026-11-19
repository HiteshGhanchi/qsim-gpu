"""Shor's period finding for N = 15, with hard-coded modular multiplication.

Registers (little-endian):
  qubits 0..7   input ("counting") register, 8 qubits -> 256-point QFT
  qubits 8..11  output register, holds y in 0..15, starts at 1

Circuit: H on the input register; for each input qubit j, multiply the output register by
a^(2^j) mod 15 controlled on that qubit; inverse QFT on the input register. Measuring the input
register gives values near multiples of 256/r, where r is the period of a^x mod 15.
Classical post-processing (continued fractions, then gcd) turns r into the factors.

The multiplication circuits are fixed swap/X networks that only work for N = 15. This shows
how the algorithm works; it is not a general factoring circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from qsim.algorithms.qft import qft_gates
from qsim.backends.base import Backend
from qsim.circuit import Circuit

N = 15
N_INPUT = 8
N_OUTPUT = 4
VALID_A = (2, 4, 7, 8, 11, 13)


def _times_a_once(c: Circuit, a: int, control: int, out: list[int]) -> None:
    """Controlled multiply of the output register by a (mod 15), as swaps and X gates."""
    o0, o1, o2, o3 = out
    if a in (2, 13):
        c.add("swap", [o2, o3], [control])
        c.add("swap", [o1, o2], [control])
        c.add("swap", [o0, o1], [control])
    if a in (7, 8):
        c.add("swap", [o0, o1], [control])
        c.add("swap", [o1, o2], [control])
        c.add("swap", [o2, o3], [control])
    if a in (4, 11):
        c.add("swap", [o1, o3], [control])
        c.add("swap", [o0, o2], [control])
    if a in (7, 11, 13):
        for q in out:
            c.cx(control, q)


def add_controlled_modmul(c: Circuit, a: int, power: int, control: int, out: list[int]) -> None:
    """Controlled y -> a^power * y mod 15. Uses a^4 = 1 (mod 15) to skip whole cycles."""
    for _ in range(power % 4):
        _times_a_once(c, a, control, out)


def shor15_circuit(a: int) -> Circuit:
    if a not in VALID_A:
        raise ValueError(f"a must be one of {VALID_A} (coprime to 15, not trivial)")
    inp = list(range(N_INPUT))
    out = list(range(N_INPUT, N_INPUT + N_OUTPUT))
    c = Circuit(N_INPUT + N_OUTPUT)
    for q in inp:
        c.h(q)
    c.x(out[0])  # output register = 1
    for j in inp:
        add_controlled_modmul(c, a, 2**j, j, out)
    c.extend(qft_gates(inp, inverse=True))
    return c


def input_distribution(probabilities: np.ndarray) -> np.ndarray:
    """Marginal probabilities of the 8-qubit input register (output qubits are the high bits)."""
    return np.asarray(probabilities).reshape(2**N_OUTPUT, 2**N_INPUT).sum(axis=0)


def period_from_measurement(measured: int, a: int) -> int | None:
    """Continued fractions: measured/256 ~ s/r. Returns r if a^r = 1 mod 15, else None."""
    if measured == 0:
        return None
    r = Fraction(measured, 2**N_INPUT).limit_denominator(N).denominator
    for candidate in (r, 2 * r, 4 * r):  # s and r may share a factor; try small multiples
        if pow(a, candidate, N) == 1:
            return candidate
    return None


def factors_from_period(a: int, r: int) -> tuple[int, int] | None:
    if r % 2:
        return None
    half = pow(a, r // 2, N)
    if half == N - 1:
        return None
    p, q = math.gcd(half - 1, N), math.gcd(half + 1, N)
    if p * q == N and 1 < p < N:
        return (min(p, q), max(p, q))
    return None


@dataclass
class ShorResult:
    a: int
    counts: dict[int, int]       # input-register value -> count
    period: int | None
    factors: tuple[int, int] | None


def run_shor15(backend: Backend, a: int, shots: int = 1024, seed: int | None = 0) -> ShorResult:
    """Run the circuit, sample the input register, and post-process classically."""
    backend.reset()
    backend.run(shor15_circuit(a))
    raw = backend.sample(shots, seed=seed)
    counts: dict[int, int] = {}
    for pattern, cnt in raw.items():
        m = pattern & (2**N_INPUT - 1)
        counts[m] = counts.get(m, 0) + cnt
    period = None
    for m, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        period = period_from_measurement(m, a)
        if period is not None:
            break
    factors = factors_from_period(a, period) if period else None
    return ShorResult(a, counts, period, factors)
