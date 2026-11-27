"""OpenQASM 2 import/export (W4) and the optional Qiskit bridge."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from qsimgpu.backends import NumpyBackend
from qsimgpu.circuit import Circuit, Gate
from qsimgpu.qasm import QasmError, QasmExportError, from_qasm2, to_qasm2
from tests.circuits import random_circuit
from tests.tolerances import ATOL_COMPLEX128

QASM_DIR = Path(__file__).parent / "data" / "qasm"
HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def state(c: Circuit) -> np.ndarray:
    unitary_part = Circuit(c.n_qubits, [g for g in c if g.name not in ("measure", "reset")])
    b = NumpyBackend(c.n_qubits)
    b.run(unitary_part)
    return b.statevector()


def assert_equal_up_to_global_phase(a: np.ndarray, b: np.ndarray) -> None:
    k = int(np.argmax(np.abs(b)))
    phase = a[k] / b[k]
    assert abs(abs(phase) - 1) < 1e-9
    np.testing.assert_allclose(a, phase * b, atol=ATOL_COMPLEX128)


# --- parsing ----------------------------------------------------------------------------

def test_basic_program() -> None:
    c = from_qasm2(HEADER + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n")
    assert [g.name for g in c] == ["h", "cx", "measure", "measure"]
    assert c.gates[1] == Gate("cx", (1,), (0,)) and c.cregs == {"c": (0, 1)}


def test_multiple_registers_are_flattened_in_order() -> None:
    c = from_qasm2(HEADER + "qreg a[2]; qreg b[1]; creg m[1]; creg n[2];\n"
                   "x b[0]; measure a[1] -> n[1];\n")
    assert c.n_qubits == 3 and c.gates[0].targets == (2,)
    assert c.gates[1].clbits == (2,) and c.cregs == {"m": (0,), "n": (1, 2)}


def test_mapped_gates_and_expressions() -> None:
    c = from_qasm2(HEADER + "qreg q[2];\nu1(pi/2) q[0]; u2(0,pi) q[1]; u3(1,2,3) q[0];\n"
                   "cu1(-pi^2/4) q[0],q[1]; crz(sqrt(2)*sin(pi/6)) q[1],q[0];\n"
                   "id q[0]; barrier q;\n")
    assert c.gates[0] == Gate("p", (0,), params=(math.pi / 2,))
    assert c.gates[1] == Gate("u", (1,), params=(math.pi / 2, 0.0, math.pi))
    assert c.gates[3] == Gate("cp", (1,), (0,), (-math.pi**2 / 4,))
    assert c.gates[4].params[0] == pytest.approx(math.sqrt(2) * 0.5)
    assert len(c) == 5  # id and barrier do nothing


def test_gate_definitions_expand_with_parameters() -> None:
    c = from_qasm2(HEADER + "qreg q[2];\ngate twist(a, b) x, y { rz(a + b) y; cx x, y; }\n"
                   "gate outer(t) p, r { twist(t, 2*t) r, p; h p; }\nouter(0.5) q[0], q[1];\n")
    assert [g.name for g in c] == ["rz", "cx", "h"]
    assert c.gates[0] == Gate("rz", (0,), params=(1.5,))
    assert c.gates[1] == Gate("cx", (0,), (1,))


def test_broadcast_and_conditions() -> None:
    c = from_qasm2(HEADER + "qreg q[3]; creg c[2];\nh q;\nif(c==2) x q[1];\nreset q;\n")
    assert [g.targets[0] for g in c.gates[:3]] == [0, 1, 2]
    assert c.gates[3].condition == ((0, 1), 2)
    assert [g.name for g in c.gates[4:]] == ["reset"] * 3


@pytest.mark.parametrize(("name", "n", "params"), [
    ("rccx", 3, ()), ("rc3x", 4, ()), ("c3x", 4, ()), ("c3sqrtx", 4, ()), ("c4x", 5, ()),
    ("cu", 2, (0.3, 0.5, -0.7, 1.1)),
])
def test_qiskit_extended_qelib1_gates_match_qiskit(name: str, n: int,
                                                   params: tuple[float, ...]) -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import library as lib
    from qiskit.quantum_info import Statevector

    gate = {"rccx": lib.RCCXGate, "rc3x": lib.RC3XGate, "c3x": lib.C3XGate,
            "c3sqrtx": lib.C3SXGate, "c4x": lib.C4XGate, "cu": lib.CUGate}[name](*params)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    v /= np.linalg.norm(v)
    qc = QuantumCircuit(n)
    qc.append(gate, list(range(n)))
    expected = np.asarray(Statevector(v).evolve(qc))
    args = ",".join(f"q[{i}]" for i in range(n))
    p = f"({','.join(map(repr, params))})" if params else ""
    c = from_qasm2(HEADER + f"qreg q[{n}];\n{name}{p} {args};\n")
    b = NumpyBackend(n)
    b.set_state(v)
    b.run(c)
    np.testing.assert_allclose(b.statevector(), expected, atol=1e-12)


def test_rzz_and_rxx_decompositions_are_exact() -> None:
    from scipy.linalg import expm

    zz = np.kron(np.diag([1, -1]), np.diag([1, -1]))
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    for name, op in (("rzz", zz), ("rxx", xx)):
        c = from_qasm2(HEADER + f"qreg q[2];\nh q[0]; ry(0.4) q[1];\n{name}(0.7) q[0],q[1];\n")
        before = state(Circuit(2, c.gates[:2]))
        assert_equal_up_to_global_phase(state(c), expm(-0.35j * op) @ before)
        np.testing.assert_allclose(state(c), expm(-0.35j * op) @ before, atol=1e-12)


@pytest.mark.parametrize(("src", "fragment", "line"), [
    ("qreg q[1];\nfoo q[0];", "gate 'foo' is not supported", 4),
    ("qreg q[1];\nh q[0]", "expected ';'", 4),
    ("qreg q[1];\nh q[1];", "index 1 out of range", 4),
    ("qreg q[2];\ncx q[0];", "cx takes 0 parameter(s) and 2 qubit(s)", 4),
    ("qreg q[1];\nrx(theta) q[0];", "unknown identifier 'theta'", 4),
    ("qreg q[1];\nrx(1/0) q[0];", "division by zero", 4),
    ("qreg a[2]; qreg b[3];\ncx a, b;", "different sizes", 4),
    ("qreg q[1];\nopaque magic q;", "opaque", 4),
    ("qreg q[2];\ncx q[0], q[0];", "same qubit", 4),
    ("qreg q[1]; creg c[1];\nif(c==2) x q[0];", "does not fit", 4),
    ("qreg q[1];\nh q[0]; @", "unexpected character '@'", 4),
])
def test_errors_point_to_the_problem(src: str, fragment: str, line: int) -> None:
    with pytest.raises(QasmError, match=re.escape(fragment)) as info:
        from_qasm2(HEADER + src)
    assert info.value.line == line


def test_header_and_include_errors() -> None:
    with pytest.raises(QasmError, match="OpenQASM 2"):
        from_qasm2("OPENQASM 3.0;\nqreg q[1];")
    with pytest.raises(QasmError, match="qelib1.inc"):
        from_qasm2('OPENQASM 2.0;\ninclude "other.inc";\nqreg q[1];')


# --- export and round trips ------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_round_trip_gives_an_identical_circuit(seed: int) -> None:
    base = random_circuit(5, 60, seed, allow_multi_controls=False)
    c = Circuit(5, list(base), n_clbits=2)
    c.u(0.1, 0.2, 0.3, 1).sx(2).sxdg(3).measure(0, 0).x(1).c_if([0], 1).reset(2).measure(1, 1)
    back = from_qasm2(to_qasm2(c))
    assert back.gates == c.gates and back.n_clbits == c.n_clbits


def test_export_refuses_what_qasm2_cannot_express() -> None:
    for c, msg in ((Circuit(1).unitary(np.eye(2), [0]), "unitary"),
                   (Circuit(4).mcz([0, 1, 2], 3), "control"),
                   (Circuit(1, n_clbits=2).x(0).c_if([0, 1], 3).x(0).c_if([0], 1), "register")):
        with pytest.raises(QasmExportError, match=msg):
            to_qasm2(c)


# --- against Qiskit Aer ------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(15))
def test_qiskit_random_circuits_match_aer(seed: int) -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit import qasm2
    from qiskit.circuit.random import random_circuit as qiskit_random
    from qiskit.quantum_info import Statevector

    qc = qiskit_random(2 + seed % 6, 12, max_operands=3, seed=seed)
    ours = state(from_qasm2(qasm2.dumps(qc)))
    # Qiskit's own custom-gate definitions can drop a global phase when written as OpenQASM 2,
    # so the comparison allows one; everything else must match.
    assert_equal_up_to_global_phase(ours, np.asarray(Statevector(qc)))


@pytest.mark.parametrize("path", sorted(QASM_DIR.glob("*.qasm")), ids=lambda p: p.stem)
def test_benchmark_files_parse_run_and_match_aer(path: Path) -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector

    text = path.read_text()
    ours = state(from_qasm2(text))
    qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    qc.remove_final_measurements()
    assert_equal_up_to_global_phase(ours, np.asarray(Statevector(qc)))


def test_qiskit_bridge_round_trip() -> None:
    pytest.importorskip("qiskit")
    from qsimgpu.interop.qiskit import from_qiskit, to_qiskit

    c = Circuit(3, list(random_circuit(3, 30, 4, allow_multi_controls=False)), n_clbits=1)
    c.measure(2, 0)
    assert from_qiskit(to_qiskit(c)).gates == c.gates
