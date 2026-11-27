"""Convert a qsim Circuit to a Qiskit QuantumCircuit (comparison only).

Qiskit uses the same little-endian order and the same gate definitions, so no reordering
or phase correction is applied here. If the two simulators disagree, that is a real bug.
"""

from __future__ import annotations

import numpy as np

from qsimgpu.circuit import Circuit


def to_qiskit(circuit: Circuit):  # type: ignore[no-untyped-def]
    from qiskit import QuantumCircuit
    from qiskit.circuit import library as lib

    one_qubit = {
        "h": lib.HGate, "x": lib.XGate, "y": lib.YGate, "z": lib.ZGate, "s": lib.SGate,
        "sdg": lib.SdgGate, "t": lib.TGate, "tdg": lib.TdgGate, "rx": lib.RXGate,
        "ry": lib.RYGate, "rz": lib.RZGate, "p": lib.PhaseGate,
    }
    base_of = {"cx": "x", "cz": "z", "cp": "p", "ccx": "x"}
    qc = QuantumCircuit(circuit.n_qubits)
    for g in circuit:
        if g.name == "unitary":
            from qiskit.circuit.library import UnitaryGate

            qc.append(UnitaryGate(np.asarray(g.matrix)), list(g.targets))
            continue
        if g.name == "swap":
            op = lib.SwapGate()
        else:
            op = one_qubit[base_of.get(g.name, g.name)](*g.params)
        if g.controls:
            op = op.control(len(g.controls), annotated=True)
        qc.append(op, list(g.controls) + list(g.targets))
    return qc


def aer_statevector(circuit: Circuit) -> np.ndarray:
    """Final statevector from Qiskit Aer (double precision)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = to_qiskit(circuit)
    qc.save_statevector()
    sim = AerSimulator(method="statevector", precision="double")
    result = sim.run(transpile(qc, sim, optimization_level=0)).result()
    return np.asarray(result.get_statevector(), dtype=np.complex128)
