"""B7: the same circuits on qsim, Qiskit Aer (CPU/GPU) and cuStateVec.

Each engine runs in the environment where it is installed (they conflict with each other):
    .venv/bin/python -m bench.run_b7_compare --engine qsim
    .venv/bin/python -m bench.run_b7_compare --engine aer-cpu
    PYTHONPATH=. <aer-gpu venv>/bin/python -m bench.run_b7_compare --engine aer-gpu
    PYTHONPATH=. <cuquantum venv>/bin/python -m bench.run_b7_compare --engine custatevec

Circuits: QFT on n qubits (H + controlled-phase + swaps) and a fixed random circuit of depth
100 drawn from the whole gate set. Timing is wall clock around one full circuit run, after a
warm-up run, with a device synchronise at the end, so it covers everything an engine needs
per circuit (kernel launches, Python overhead). Circuit construction and Aer transpilation
happen before the clock starts. Median of --repeats runs.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from bench.run_env import save_result
from qsim.algorithms.qft import qft
from qsim.circuit import Circuit
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

def engine_qsim(c: Circuit) -> Callable[[], Any]:
    from qsim.backends.cuda_backend import CudaBackend

    b = CudaBackend(c.n_qubits, np.complex64)

    def run() -> None:
        b.reset()
        b.run(c)
        b.synchronize()
    return run


def engine_aer(c: Circuit, device: str) -> Callable[[], Any]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from tests.qiskit_bridge import to_qiskit

    sim = AerSimulator(method="statevector", device=device, precision="single")
    qc = to_qiskit(c)
    qc.save_statevector()
    tqc = transpile(qc, sim, optimization_level=0)

    def run() -> None:
        sim.run(tqc).result()
    return run


def engine_custatevec(c: Circuit) -> Callable[[], Any]:
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

    def run() -> None:
        sv.fill(0)
        sv[0] = 1
        for m, targets, controls in prepared:
            cusv.apply_matrix(handle, sv.data.ptr, c32, c.n_qubits, m.ctypes.data, c32, row, 0,
                              targets.ctypes.data, len(targets),
                              controls.ctypes.data if len(controls) else 0, 0, len(controls),
                              cusv.ComputeType.COMPUTE_32F, 0, 0)
        cp.cuda.Device().synchronize()

    run.state = sv  # type: ignore[attr-defined]
    return run


def check_custatevec_matches_oracle() -> float:
    """Make sure the cuStateVec calls above implement the same circuit."""
    from qsim.backends import NumpyBackend

    c = random_circuit(8, 60, seed=7)
    run = engine_custatevec(c)
    run()
    ref = NumpyBackend(8)
    ref.run(c)
    return float(np.max(np.abs(run.state.get() - ref.statevector())))  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["qsim", "aer-cpu", "aer-gpu", "custatevec"],
                    required=True)
    ap.add_argument("--qubits", default="10,14,18,20,22,24")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    extra: dict[str, Any] = {}
    if args.engine == "custatevec":
        extra["max_abs_diff_vs_oracle_n8"] = check_custatevec_matches_oracle()
        print(f"cuStateVec vs oracle (n=8): {extra['max_abs_diff_vs_oracle_n8']:.2e}")
    rows = []
    for n in [int(x) for x in args.qubits.split(",")]:
        for name, c in circuits(n).items():
            if args.engine == "qsim":
                fn = engine_qsim(c)
            elif args.engine == "custatevec":
                fn = engine_custatevec(c)
            else:
                fn = engine_aer(c, "CPU" if args.engine == "aer-cpu" else "GPU")
            t = time_median(fn, args.repeats)
            rows.append({"circuit": name, "n": n, "gates": len(c), "timing": t})
            print(f"{args.engine:10s} {name:12s} n={n:2d} gates={len(c):4d} "
                  f"{t['median_ms']:10.2f} ms")
    path = save_result(f"b7_compare_{args.engine}", {"engine": args.engine, "dtype": "complex64",
                                                     "rows": rows, **extra})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
