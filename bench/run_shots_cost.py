"""W3: cost of run_shots with and without mid-circuit measurement.

    python -m bench.run_shots_cost

Same 12-qubit, 60-gate random circuit measured at the end (fast path: one run, then sample)
and with one extra mid-circuit measurement (every shot re-runs the circuit).
"""

from __future__ import annotations

import time

from bench.run_env import save_result
from qsimgpu.backends import NumpyBackend
from qsimgpu.backends.cuda_backend import CudaBackend
from qsimgpu.circuit import Circuit
from qsimgpu.execution import run_shots
from tests.circuits import random_circuit


def main() -> None:
    n, shots = 12, 1000
    base = random_circuit(n, 60, seed=11)
    end = Circuit(n, list(base), n_clbits=n)
    for q in range(n):
        end.measure(q, q)
    mid = Circuit(n, list(base), n_clbits=n)
    mid.measure(0, 0).h(0)
    for q in range(n):
        mid.measure(q, q)
    rows = []
    for name, make in (("numpy", NumpyBackend), ("cuda", CudaBackend)):
        cases = (("measure at end (sample once)", end), ("mid-circuit (re-run per shot)", mid))
        for label, c in cases:
            b = make(n)
            run_shots(b, c, 10, seed=0)  # warm-up
            t0 = time.perf_counter()
            run_shots(b, c, shots, seed=1)
            ms = (time.perf_counter() - t0) * 1e3
            rows.append({"backend": name, "case": label, "shots": shots, "ms": ms})
            print(f"{name:6s} {label:32s} {shots} shots: {ms:9.1f} ms")
    path = save_result("w3_shots", {"n": n, "gates": len(base), "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
