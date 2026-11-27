"""Running circuits that contain measurement, reset and classical conditions.

`run_circuit` is the one loop every backend's `run` uses, so the NumPy and CUDA backends
treat classical bits identically:
- unitary gates go to backend.apply;
- a gate with a condition is applied only if the classical register value matches;
- measure draws ONE uniform number from the random generator and asks the backend for
  P(qubit = 1); the outcome is 1 if the number is below it. Because both backends consume the
  generator in the same order, the same seed gives the same bits on both. The only exception is
  a draw that lands within the precision gap of the two backends (about 1e-7 for complex64)
  of the threshold, which happens with roughly that probability per measurement;
- reset is a measurement followed by an X if the result was 1.

`run_shots` repeats a circuit. If all measurements are at the end and there is no reset or
condition, the state is prepared once and the fast `sample()` draws every shot (cost: one run).
Otherwise each shot re-runs the whole circuit from |0...0> (cost: shots x one run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qsimgpu.circuit import Circuit, Gate

if TYPE_CHECKING:
    from qsimgpu.backends.base import Backend, StepCallback

RngLike = np.random.Generator | int | None


def as_generator(rng: RngLike) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def register_value(bits: list[int], clbits: tuple[int, ...]) -> int:
    """Value of the register formed by `clbits`; bit j of the value is clbits[j]."""
    return sum(bits[c] << j for j, c in enumerate(clbits))


def run_circuit(backend: Backend, circuit: Circuit, on_step: StepCallback | None,
                rng: RngLike) -> tuple[int, ...]:
    """Apply `circuit` to `backend`'s current state; return the final classical bits."""
    if circuit.n_qubits != backend.n_qubits:
        raise ValueError(f"circuit has {circuit.n_qubits} qubits, backend {backend.n_qubits}")
    bits = [0] * circuit.n_clbits
    backend.classical_bits = tuple(bits)
    gen: np.random.Generator | None = None
    for i, g in enumerate(circuit):
        if g.condition is None or register_value(bits, g.condition[0]) == g.condition[1]:
            if g.name == "measure":
                gen = gen or as_generator(rng)
                bits[g.clbits[0]] = backend.measure_qubit(g.targets[0], gen)
                backend.classical_bits = tuple(bits)
            elif g.name == "reset":
                gen = gen or as_generator(rng)
                backend.reset_qubit(g.targets[0], gen)
            else:
                backend.apply(_strip_condition(g))
        if on_step is not None:
            on_step(i, g, backend)
    return tuple(bits)


def _strip_condition(g: Gate) -> Gate:
    if g.condition is None:
        return g
    return Gate(g.name, g.targets, g.controls, g.params, g.matrix)


def collapse_outcome(p_one: float, gen: np.random.Generator) -> int:
    """Draw the measurement outcome from P(1). Always consumes exactly one random number."""
    return 1 if gen.random() < p_one else 0


def has_mid_circuit_measurement(circuit: Circuit) -> bool:
    """True unless every measurement comes after all other gates and there is no reset or
    condition (then the state can be prepared once and sampled)."""
    seen_measure = False
    for g in circuit:
        if g.name == "reset" or g.condition is not None:
            return True
        if g.name == "measure":
            seen_measure = True
        elif seen_measure:
            return True
    return False


def run_shots(backend: Backend, circuit: Circuit, shots: int,
              seed: int | None = None) -> dict[int, int]:
    """Run `circuit` `shots` times from |0...0>; return {classical register value: count}.

    The register value has bit j = clbit j. Circuits without mid-circuit measurement use one
    run plus `sample()`; the others are re-run once per shot.
    """
    if shots < 0:
        raise ValueError("shots must be >= 0")
    if not has_mid_circuit_measurement(circuit):
        unitary = Circuit(circuit.n_qubits, [g for g in circuit if g.name != "measure"])
        where = [(g.targets[0], g.clbits[0]) for g in circuit if g.name == "measure"]
        backend.reset()
        backend.run(unitary)
        counts: dict[int, int] = {}
        for pattern, n in backend.sample(shots, seed=seed).items():
            bits = [0] * circuit.n_clbits
            for q, c in where:
                bits[c] = (pattern >> q) & 1
            key = register_value(bits, tuple(range(circuit.n_clbits)))
            counts[key] = counts.get(key, 0) + n
        return counts
    gen = as_generator(seed)
    counts = {}
    for _ in range(shots):
        backend.reset()
        bits = list(backend.run(circuit, rng=gen))
        key = register_value(bits, tuple(range(circuit.n_clbits)))
        counts[key] = counts.get(key, 0) + 1
    return counts
