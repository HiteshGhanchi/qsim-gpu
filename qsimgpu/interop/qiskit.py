"""Convert between qsimgpu circuits and Qiskit (optional: needs `pip install qsim-gpu[compare]`).

    from qsimgpu.interop.qiskit import to_qiskit, from_qiskit, from_qasm3

Qiskit uses the same little-endian qubit order and the same gate definitions, so conversions
apply no reordering or phase correction.
- to_qiskit: every qsimgpu gate, including `unitary`, measure, reset, and conditions on a single
  clbit or on a whole named classical register (as `if_test` blocks).
- from_qiskit: goes through OpenQASM 2 (`qiskit.qasm2.dumps` then `from_qasm2`), so it supports
  whatever Qiskit can write as OpenQASM 2, including custom gate definitions.
- from_qasm3: OpenQASM 3 through `qiskit.qasm3.loads` (which itself needs the
  `qiskit-qasm3-import` package). Optional; the hand-written parser covers OpenQASM 2 only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qsimgpu.circuit import Circuit
from qsimgpu.qasm import from_qasm2


def _require_qiskit() -> Any:
    try:
        import qiskit
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("this needs qiskit: pip install 'qsim-gpu[compare]'") from exc
    return qiskit


def to_qiskit(circuit: Circuit) -> Any:
    """A qiskit.QuantumCircuit equivalent to `circuit`."""
    qiskit = _require_qiskit()
    from qiskit.circuit import library as lib

    one_qubit = {
        "h": lib.HGate, "x": lib.XGate, "y": lib.YGate, "z": lib.ZGate, "s": lib.SGate,
        "sdg": lib.SdgGate, "t": lib.TGate, "tdg": lib.TdgGate, "rx": lib.RXGate,
        "ry": lib.RYGate, "rz": lib.RZGate, "p": lib.PhaseGate, "sx": lib.SXGate,
        "sxdg": lib.SXdgGate, "u": lib.UGate,
    }
    base_of = {"cx": "x", "cz": "z", "cp": "p", "ccx": "x"}
    qr = qiskit.QuantumRegister(circuit.n_qubits, "q")
    regs = {name: qiskit.ClassicalRegister(len(bits), name) for name, bits in circuit.cregs.items()}
    if circuit.n_clbits and not regs:
        regs = {"c": qiskit.ClassicalRegister(circuit.n_clbits, "c")}
        bits_of = {"c": tuple(range(circuit.n_clbits))}
    else:
        bits_of = dict(circuit.cregs)
    qc = qiskit.QuantumCircuit(qr, *regs.values())
    clbit = {b: regs[name][k] for name, bits in bits_of.items() for k, b in enumerate(bits)}
    by_bits = {bits: regs[name] for name, bits in bits_of.items()}

    def add(g: Any) -> None:
        if g.name == "measure":
            qc.measure(g.targets[0], clbit[g.clbits[0]])
        elif g.name == "reset":
            qc.reset(g.targets[0])
        elif g.name == "unitary":
            qc.append(lib.UnitaryGate(np.asarray(g.matrix)), list(g.targets))
        else:
            op = lib.SwapGate() if g.name == "swap" else one_qubit[base_of.get(g.name, g.name)](
                *g.params)
            if g.controls:
                op = op.control(len(g.controls), annotated=True)
            qc.append(op, list(g.controls) + list(g.targets))

    for g in circuit:
        if g.condition is None:
            add(g)
            continue
        bits, value = g.condition
        if len(bits) == 1:
            target: Any = clbit[bits[0]]
        elif bits in by_bits:
            target = by_bits[bits]
        else:
            raise ValueError(f"condition on clbits {bits} is neither one bit nor a named register")
        with qc.if_test((target, value)):
            add(g)
    return qc


def from_qiskit(qc: Any) -> Circuit:
    """A qsimgpu Circuit from a qiskit.QuantumCircuit (via OpenQASM 2)."""
    _require_qiskit()
    from qiskit import qasm2

    return from_qasm2(qasm2.dumps(qc))


def from_qasm3(text: str) -> Circuit:
    """Parse OpenQASM 3 through Qiskit (needs qiskit and qiskit-qasm3-import)."""
    _require_qiskit()
    from qiskit import qasm3

    return from_qiskit(qasm3.loads(text))
