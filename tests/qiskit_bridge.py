"""Test helpers for the Aer cross-checks. The conversion itself lives in qsimgpu.interop.qiskit."""

from __future__ import annotations

import numpy as np

from qsimgpu.circuit import Circuit
from qsimgpu.interop.qiskit import to_qiskit

__all__ = ["aer_statevector", "to_qiskit"]


def aer_statevector(circuit: Circuit) -> np.ndarray:
    """Final statevector from Qiskit Aer (double precision)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = to_qiskit(circuit)
    qc.save_statevector()
    sim = AerSimulator(method="statevector", precision="double")
    result = sim.run(transpile(qc, sim, optimization_level=0)).result()
    return np.asarray(result.get_statevector(), dtype=np.complex128)
