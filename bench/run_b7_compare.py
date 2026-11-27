"""B7: the same circuits on qsim, Qiskit Aer (CPU/GPU) and cuStateVec.

Each engine runs in the environment where it is installed (they conflict with each other):
    .venv/bin/python -m bench.run_b7_compare --engine qsim            (also: qsim-fused)
    .venv/bin/python -m bench.run_b7_compare --engine aer-cpu         (also: aer-cpu-nofusion)
    PYTHONPATH=. <aer-gpu venv>/bin/python -m bench.run_b7_compare --engine aer-gpu
                                                                      (also: aer-gpu-nofusion)
    PYTHONPATH=. <cuquantum venv>/bin/python -m bench.run_b7_compare --engine custatevec

Fairness rules (every engine follows all of them):
- Same circuits: QFT on n qubits and one fixed random circuit of depth 100 from the whole
  gate set. Same precision: single (complex64).
- Same output: each run starts from |0...0>, applies the circuit, and returns ONE number to
  Python, <Z> on qubit 0. No engine copies the statevector back. (Asking Aer for the whole
  statevector added ~16% to its GPU time at 24 qubits, while qsim returned nothing, which
  favoured qsim. This rule removes that.)
- Same state size: Aer's qubit truncation is switched off, so every engine simulates all n
  qubits.
- Gate fusion: Aer fuses by default, so it is timed with and without fusion (`-nofusion`).
  qsim is timed without fusion and with it (`qsim-fused`, m = 2, the best width in B5);
  the fusion pass runs inside the timed region, as Aer's does. cuStateVec is driven gate
  by gate through its general apply_matrix call, with no fusion.
- Timing: wall clock around one full run, after one warm-up run, median of --repeats (7).
  Circuit construction and Aer transpilation happen before the clock starts.
The <Z0> values of all engines are stored, so they can be checked against each other.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from bench.run_env import save_result
from qsimgpu.algorithms.qft import qft
from qsimgpu.circuit import Circuit
from tests.circuits import random_circuit


def circuits(n: int) -> dict[str, Circuit]:
    return {"qft": qft(n), "random_d100": random_circuit(n, 100, seed=2024)}


def time_median(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    fn()  # warm-up (compiles kernels, allocates)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts),
            "all_ms": ts}


# --- engines -------------------------------------------------------------------------

def _z0(n: int) -> str:
    return "I" * (n - 1) + "Z"


def engine_qsim(c: Circuit, fused: bool = False) -> Callable[[], float]:
    from qsimgpu.backends.cuda_backend import CudaBackend
    from qsimgpu.passes.fusion import fuse

    b = CudaBackend(c.n_qubits, np.complex64)

    def run() -> float:
        b.reset()
        b.run(fuse(c, 2) if fused else c)
        return b.expectation_pauli(_z0(c.n_qubits))  # returns to the host, so it synchronises
    return run


def engine_aer(c: Circuit, device: str, fusion: bool) -> Callable[[], float]:
    from qiskit import transpile
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator

    from tests.qiskit_bridge import to_qiskit

    # enable_truncation=False: by default Aer drops qubits it judges inactive (it simulated
    # only 21 of 24 qubits of the random circuit), which would no longer be the same job.
    sim = AerSimulator(method="statevector", device=device, precision="single",
                       fusion_enable=fusion, enable_truncation=False)
    qc = to_qiskit(c)
    qc.save_expectation_value(Pauli("Z"), [0])
    tqc = transpile(qc, sim, optimization_level=0)

    def run() -> float:
        return float(np.real(sim.run(tqc).result().data(0)["expectation_value"]))
    return run


def engine_custatevec(c: Circuit) -> Callable[[], float]:
    import cupy as cp

    try:
        from cuquantum.bindings import custatevec as cusv
    except ImportError:  # older cuquantum layout
        from cuquantum import custatevec as cusv

    handle = cusv.create()
    sv = cp.zeros(1 << c.n_qubits, dtype=cp.complex64)
    prepared = []
    for g in c:
        m = np.ascontiguousarray(g.target_matrix(), dtype=np.complex64)
        targets = np.asarray(g.targets, dtype=np.int32)
        controls = np.asarray(g.controls, dtype=np.int32)
        prepared.append((m, targets, controls))
    c32, row = 4, 1  # CUDA_C_32F, row-major layout
    basis = np.array([0], dtype=np.int32)

    def run() -> float:
        sv.fill(0)
        sv[0] = 1
        for m, targets, controls in prepared:
            cusv.apply_matrix(handle, sv.data.ptr, c32, c.n_qubits, m.ctypes.data, c32, row, 0,
                              targets.ctypes.data, len(targets),
                              controls.ctypes.data if len(controls) else 0, 0, len(controls),
                              cusv.ComputeType.COMPUTE_32F, 0, 0)
        # probability mass with qubit 0 = 0 and = 1, computed on the GPU, returned to Python
        p0, p1 = cusv.abs2sum_on_z_basis(handle, sv.data.ptr, c32, c.n_qubits, True, True,
                                         basis.ctypes.data, 1)
        return float(p0 - p1)

    run.state = sv  # type: ignore[attr-defined]
    return run


def check_custatevec_matches_oracle() -> float:
    """Make sure the cuStateVec calls above implement the same circuit."""
    from qsimgpu.backends import NumpyBackend

    c = random_circuit(8, 60, seed=7)
    run = engine_custatevec(c)
    run()
    ref = NumpyBackend(8)
    ref.run(c)
    return float(np.max(np.abs(run.state.get() - ref.statevector())))  # type: ignore[attr-defined]


ENGINES = ["qsim", "qsim-fused", "aer-cpu", "aer-cpu-nofusion", "aer-gpu", "aer-gpu-nofusion",
           "custatevec"]


def make_engine(engine: str, c: Circuit) -> Callable[[], float]:
    if engine in ("qsim", "qsim-fused"):
        return engine_qsim(c, fused=engine == "qsim-fused")
    if engine == "custatevec":
        return engine_custatevec(c)
    device = "CPU" if engine.startswith("aer-cpu") else "GPU"
    return engine_aer(c, device, fusion=not engine.endswith("nofusion"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ENGINES, required=True)
    ap.add_argument("--qubits", default="10,14,18,20,22,24")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    extra: dict[str, Any] = {}
    if args.engine == "custatevec":
        extra["max_abs_diff_vs_oracle_n8"] = check_custatevec_matches_oracle()
        print(f"cuStateVec vs oracle (n=8): {extra['max_abs_diff_vs_oracle_n8']:.2e}")
    rows = []
    for n in [int(x) for x in args.qubits.split(",")]:
        for name, c in circuits(n).items():
            fn = make_engine(args.engine, c)
            z0 = fn()
            t = time_median(fn, args.repeats)
            rows.append({"circuit": name, "n": n, "gates": len(c), "z0": z0, "timing": t})
            print(f"{args.engine:16s} {name:12s} n={n:2d} gates={len(c):4d} "
                  f"{t['median_ms']:10.2f} ms   <Z0>={z0:+.6f}")
    path = save_result(f"b7_compare_{args.engine}", {
        "engine": args.engine, "dtype": "complex64", "observable": "Z on qubit 0",
        "repeats": args.repeats, "rows": rows, **extra})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
