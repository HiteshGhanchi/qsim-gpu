"""Circuit IR: a circuit is plain data, a list of Gate records.

Keeping "what to compute" (this file) apart from "how to run it" (the backends) is what
lets the NumPy reference and the CUDA backend run identical circuits, and lets the fusion
pass rewrite a circuit before any backend sees it.

Canonical encoding of controlled gates: `targets` holds only the target qubit(s) and
`controls` holds the control qubit(s). For example CNOT with control 0 and target 1 is
Gate("cx", targets=(1,), controls=(0,)). Any 1-qubit gate may also carry controls
(a generic controlled gate), e.g. Gate("z", targets=(3,), controls=(0, 1, 2)).

Classical bits: a circuit may have n_clbits classical bits. Gate("measure", (q,), clbits=(c,))
measures qubit q into bit c, collapsing the state; Gate("reset", (q,)) puts qubit q back to |0>.
Any gate may carry a condition (clbits, value): it is applied only if the register formed by
those bits equals value, with bit j of value belonging to clbits[j] (little-endian, like qubits).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from qsimgpu import gates
from qsimgpu.gates import Matrix


@dataclass(frozen=True, eq=False)
class Gate:
    """One operation. Validated on creation; qubit indices are checked by Circuit."""

    name: str
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    matrix: Matrix | None = field(default=None, repr=False)  # only for name == "unitary"
    clbits: tuple[int, ...] = ()  # only for "measure": where the result goes
    condition: tuple[tuple[int, ...], int] | None = None  # (clbits, value): apply only if equal

    def __post_init__(self) -> None:
        if self.name not in gates.SPECS:
            raise ValueError(f"unknown gate {self.name!r}")
        spec = gates.SPECS[self.name]
        # Normalise containers so equal gates compare equal however they were built.
        object.__setattr__(self, "targets", tuple(int(q) for q in self.targets))
        object.__setattr__(self, "controls", tuple(int(q) for q in self.controls))
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))
        object.__setattr__(self, "clbits", tuple(int(c) for c in self.clbits))
        if self.name == "measure" and len(self.clbits) != 1:
            raise ValueError(f"measure needs exactly one clbit, got {self.clbits}")
        if self.name != "measure" and self.clbits:
            raise ValueError(f"only 'measure' writes clbits, not {self.name!r}")
        if self.condition is not None:
            bits, value = self.condition
            bits = tuple(int(c) for c in bits)
            if not bits or len(set(bits)) != len(bits) or any(c < 0 for c in bits):
                raise ValueError(f"condition needs distinct, non-negative clbits, got {bits}")
            if not 0 <= int(value) < (1 << len(bits)):
                raise ValueError(f"condition value {value} does not fit in {len(bits)} bit(s)")
            object.__setattr__(self, "condition", (bits, int(value)))

        if spec.n_targets is not None and len(self.targets) != spec.n_targets:
            raise ValueError(f"{self.name} needs {spec.n_targets} target(s), got {self.targets}")
        if not self.targets:
            raise ValueError(f"{self.name} needs at least one target")
        if spec.n_controls is not None and len(self.controls) != spec.n_controls:
            raise ValueError(f"{self.name} needs {spec.n_controls} control(s), got {self.controls}")
        if len(self.params) != spec.n_params:
            raise ValueError(f"{self.name} needs {spec.n_params} parameter(s), got {self.params}")
        qubits = self.targets + self.controls
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"{self.name}: targets and controls must be distinct, got {qubits}")
        if any(q < 0 for q in qubits):
            raise ValueError(f"{self.name}: qubit indices must be >= 0, got {qubits}")

        if self.name == "unitary":
            if self.matrix is None:
                raise ValueError("unitary gate needs a matrix")
            m = np.asarray(self.matrix, dtype=np.complex128)
            dim = 1 << len(self.targets)
            if m.shape != (dim, dim):
                raise ValueError(
                    f"matrix shape {m.shape} does not match {len(self.targets)} targets")
            if not gates.is_unitary(m, atol=1e-8):
                raise ValueError("matrix is not unitary")
            m.setflags(write=False)
            object.__setattr__(self, "matrix", m)
        elif self.matrix is not None:
            raise ValueError(f"only 'unitary' gates take a matrix, not {self.name!r}")

    @property
    def qubits(self) -> tuple[int, ...]:
        """All qubits this gate touches: targets first, then controls."""
        return self.targets + self.controls

    @property
    def kind(self) -> gates.Kind:
        return gates.SPECS[self.name].kind

    @property
    def is_unitary(self) -> bool:
        """False for measure/reset and for any gate with a classical condition."""
        return self.name not in gates.NONUNITARY and self.condition is None

    def with_condition(self, clbits: Sequence[int], value: int) -> Gate:
        return Gate(self.name, self.targets, self.controls, self.params, self.matrix,
                    self.clbits, (tuple(clbits), value))

    def target_matrix(self) -> Matrix:
        """Matrix applied to the targets when every control is 1."""
        if self.name in gates.NONUNITARY:
            raise ValueError(f"{self.name} has no matrix; Backend.run handles it")
        if self.matrix is not None:
            return np.array(self.matrix, dtype=np.complex128)
        return gates.base_matrix(gates.SPECS[self.name].base, self.params)

    def inverse(self) -> Gate:
        if not self.is_unitary:
            raise ValueError(f"{self.name} (with its condition, if any) cannot be inverted")
        if self.name == "unitary":
            assert self.matrix is not None
            return Gate("unitary", self.targets, matrix=self.matrix.conj().T)
        name, params = gates.inverse_name_params(self.name, self.params)
        return Gate(name, self.targets, self.controls, params)

    # A frozen dataclass with an ndarray field can't use the generated __eq__/__hash__
    # (comparing arrays with == gives an array, not a bool), so define them by hand.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        same = (self.name, self.targets, self.controls, self.params, self.clbits,
                self.condition) == (other.name, other.targets, other.controls, other.params,
                                    other.clbits, other.condition)
        if not same:
            return False
        if self.matrix is None or other.matrix is None:
            return self.matrix is None and other.matrix is None
        return bool(np.array_equal(self.matrix, other.matrix))

    def __hash__(self) -> int:
        return hash((self.name, self.targets, self.controls, self.params, self.clbits,
                     self.condition))


class Circuit:
    """An ordered list of gates on n_qubits qubits, with builder methods.

    Builder methods return the circuit, so calls can be chained:
        Circuit(2).h(0).cx(0, 1)
    """

    def __init__(self, n_qubits: int, gate_list: Sequence[Gate] = (), n_clbits: int = 0) -> None:
        if n_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")
        if n_clbits < 0:
            raise ValueError("n_clbits must be >= 0")
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.gates: list[Gate] = []
        for g in gate_list:
            self.append(g)

    # --- core -------------------------------------------------------------
    def append(self, gate: Gate) -> Circuit:
        bad = [q for q in gate.qubits if q >= self.n_qubits]
        if bad:
            raise ValueError(f"{gate.name}: qubit(s) {bad} out of range for {self.n_qubits} qubits")
        used = gate.clbits + (gate.condition[0] if gate.condition else ())
        bad = [c for c in used if c >= self.n_clbits]
        if bad:
            raise ValueError(f"{gate.name}: clbit(s) {bad} out of range for {self.n_clbits} clbits")
        self.gates.append(gate)
        return self

    def measure(self, qubit: int, clbit: int) -> Circuit:
        """Measure qubit into clbit (collapses the state)."""
        return self.append(Gate("measure", (qubit,), clbits=(clbit,)))

    def reset(self, qubit: int) -> Circuit:
        """Put qubit back to |0> (measure it, then flip it if the result was 1)."""
        return self.append(Gate("reset", (qubit,)))

    def c_if(self, clbits: Sequence[int], value: int) -> Circuit:
        """Make the most recently added gate conditional: apply it only if the classical
        register formed by `clbits` (bit j of value = clbits[j]) equals `value`."""
        if not self.gates:
            raise ValueError("c_if needs a gate to condition")
        last = self.gates.pop()
        try:
            return self.append(last.with_condition(clbits, value))
        except ValueError:
            self.gates.append(last)
            raise

    @property
    def is_unitary(self) -> bool:
        return all(g.is_unitary for g in self.gates)

    def add(self, name: str, targets: Sequence[int], controls: Sequence[int] = (),
            params: Sequence[float] = ()) -> Circuit:
        return self.append(Gate(name, tuple(targets), tuple(controls), tuple(params)))

    def extend(self, other: Circuit | Sequence[Gate]) -> Circuit:
        for g in other:
            self.append(g)
        return self

    def __iter__(self) -> Iterator[Gate]:
        return iter(self.gates)

    def __len__(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, gates={len(self.gates)})"

    # --- single-qubit builders -------------------------------------------
    def h(self, q: int) -> Circuit:
        return self.add("h", [q])

    def x(self, q: int) -> Circuit:
        return self.add("x", [q])

    def y(self, q: int) -> Circuit:
        return self.add("y", [q])

    def z(self, q: int) -> Circuit:
        return self.add("z", [q])

    def s(self, q: int) -> Circuit:
        return self.add("s", [q])

    def sdg(self, q: int) -> Circuit:
        return self.add("sdg", [q])

    def t(self, q: int) -> Circuit:
        return self.add("t", [q])

    def tdg(self, q: int) -> Circuit:
        return self.add("tdg", [q])

    def rx(self, theta: float, q: int) -> Circuit:
        return self.add("rx", [q], params=[theta])

    def ry(self, theta: float, q: int) -> Circuit:
        return self.add("ry", [q], params=[theta])

    def rz(self, theta: float, q: int) -> Circuit:
        return self.add("rz", [q], params=[theta])

    def p(self, phi: float, q: int) -> Circuit:
        return self.add("p", [q], params=[phi])

    # --- multi-qubit builders (argument order follows Qiskit: controls first) ----
    def cx(self, control: int, target: int) -> Circuit:
        return self.add("cx", [target], [control])

    def cz(self, control: int, target: int) -> Circuit:
        return self.add("cz", [target], [control])

    def cp(self, phi: float, control: int, target: int) -> Circuit:
        return self.add("cp", [target], [control], [phi])

    def swap(self, a: int, b: int) -> Circuit:
        return self.add("swap", [a, b])

    def ccx(self, c1: int, c2: int, target: int) -> Circuit:
        return self.add("ccx", [target], [c1, c2])

    def mcx(self, controls: Sequence[int], target: int) -> Circuit:
        """X on target when every control is 1 (any number of controls)."""
        return self.add("x", [target], controls)

    def mcz(self, controls: Sequence[int], target: int) -> Circuit:
        """Z on target when every control is 1: flips the sign of the all-ones pattern."""
        return self.add("z", [target], controls)

    def unitary(self, matrix: Matrix, targets: Sequence[int]) -> Circuit:
        return self.append(Gate("unitary", tuple(targets), matrix=matrix))

    # --- whole-circuit operations ----------------------------------------
    def inverse(self) -> Circuit:
        """The circuit that undoes this one: reversed order, each gate inverted.

        Raises ValueError if the circuit contains measure, reset or conditional gates.
        """
        return Circuit(self.n_qubits, [g.inverse() for g in reversed(self.gates)],
                       n_clbits=self.n_clbits)

    def depth(self) -> int:
        """Number of layers when every gate starts as early as its qubits allow."""
        level = [0] * self.n_qubits
        for g in self.gates:
            start = max(level[q] for q in g.qubits)
            for q in g.qubits:
                level[q] = start + 1
        return max(level, default=0)

    def count_ops(self) -> dict[str, int]:
        """How many gates of each name, for benchmark reports."""
        return dict(Counter(g.name for g in self.gates))
