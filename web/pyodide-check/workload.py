"""The same workload in CPython and in Pyodide, so their outputs can be compared number by number.

Used by web/pyodide-check/check.mjs (Node + Pyodide), index.html (a browser) and
scripts/compare_pyodide.py (CPython reference). Needs only qsimgpu and NumPy.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import Any

import numpy as np

import qsimgpu
from qsimgpu import Circuit, NumpyBackend, views


def trace_circuit() -> Circuit:
    """A fixed 15-gate, 10-qubit circuit touching several gate families."""
    c = Circuit(10)
    for q in range(5):
        c.h(q)
    c.cx(0, 5).cx(1, 6).rz(0.3, 2).ry(-0.7, 3).cp(0.5, 4, 9)
    c.swap(2, 8).ccx(0, 1, 7).t(5).sx(6).u(0.1, 0.2, 0.3, 9)
    return c


def results() -> dict[str, Any]:
    bell = NumpyBackend(2)
    bell.run(Circuit(2).h(0).cx(0, 1))
    sv = bell.statevector()
    qasm = qsimgpu.from_qasm2('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\n'
                              "cx q[0],q[1];\ncx q[1],q[2];\n")
    ghz = NumpyBackend(3)
    ghz.run(qasm)
    return {
        "bell": [[float(z.real), float(z.imag)] for z in sv],
        "ghz_probabilities": [float(p) for p in ghz.probabilities()],
        "trace": views.step_trace(NumpyBackend(10), trace_circuit()),
    }


def timings(sizes: tuple[int, ...] = (10, 14, 18), repeats: int = 3) -> list[dict[str, Any]]:
    """Wall-clock time per gate of the NumPy backend for an n-qubit layered circuit."""
    rows = []
    for n in sizes:
        c = Circuit(n)
        for layer in range(2):
            for q in range(n):
                c.h(q).rz(0.1 * (q + layer), q)
            for q in range(n - 1):
                c.cx(q, q + 1)
        b = NumpyBackend(n)
        ts = []
        for _ in range(repeats):
            b.reset()
            t0 = time.perf_counter()
            b.run(c)
            ts.append(time.perf_counter() - t0)
        ms = sorted(ts)[len(ts) // 2] * 1e3
        rows.append({"n": n, "gates": len(c), "ms_total": ms, "ms_per_gate": ms / len(c)})
    return rows


def info() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "qsimgpu": qsimgpu.__version__, "platform": platform.platform()}


def main_json(with_timings: bool = False) -> str:
    out: dict[str, Any] = {"info": info(), "results": results()}
    if with_timings:
        out["timings"] = timings()
    return json.dumps(out)
