"""B6: complex64 vs complex128 - accuracy, speed and memory.

    python -m bench.run_b6_precision
"""

from __future__ import annotations

import argparse

import cupy as cp
import numpy as np

from bench.run_env import save_result, time_gpu
from qsimgpu import gates
from qsimgpu.backends import NumpyBackend
from qsimgpu.backends.cuda_backend import CudaBackend
from tests.circuits import random_circuit

DTYPES = {"complex64": np.complex64, "complex128": np.complex128}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-qubits", default="10,14,18,20")
    ap.add_argument("--time-qubits", default="20,22,24,26")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    accuracy = []
    for n in [int(x) for x in args.error_qubits.split(",")]:
        row: dict[str, object] = {"n": n}
        for seed in range(3):
            c = random_circuit(n, 100, 5000 + 10 * n + seed)
            ref = NumpyBackend(n)
            ref.run(c)
            for name, dt in DTYPES.items():
                b = CudaBackend(n, dt)
                b.run(c)
                err = float(np.max(np.abs(b.statevector().astype(np.complex128)
                                          - ref.statevector())))
                key = f"{name}_max_abs_error"
                row[key] = max(err, float(row.get(key, 0.0)))  # type: ignore[arg-type]
        accuracy.append(row)
        print(f"n={n:2d}  complex64 err {row['complex64_max_abs_error']:.2e}   "
              f"complex128 err {row['complex128_max_abs_error']:.2e}")

    h = gates.base_matrix("h")
    speed = []
    for n in [int(x) for x in args.time_qubits.split(",")]:
        row = {"n": n}
        for name, dt in DTYPES.items():
            b = CudaBackend(n, dt)
            t = time_gpu(lambda b=b, n=n: [b.apply_1q_matrix(h, n // 2) for _ in range(3)],
                         repeats=args.repeats)
            row[f"{name}_ms_per_gate"] = t["median_ms"] / 3
            del b
            cp.get_default_memory_pool().free_all_blocks()
        row["ratio_128_over_64"] = row["complex128_ms_per_gate"] / row["complex64_ms_per_gate"]  # type: ignore[operator]
        speed.append(row)
        print(f"n={n:2d}  complex64 {row['complex64_ms_per_gate']:.3f} ms  "
              f"complex128 {row['complex128_ms_per_gate']:.3f} ms  "
              f"ratio {row['ratio_128_over_64']:.2f}")

    total = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"]
    memory = {name: {"bytes_per_amplitude": np.dtype(dt).itemsize,
                     "max_qubits_state_only": int(np.floor(np.log2(total / np.dtype(dt).itemsize)))}
              for name, dt in DTYPES.items()}
    print("max qubits by memory alone (state only):",
          {k: v["max_qubits_state_only"] for k, v in memory.items()})
    path = save_result("b6_precision", {"accuracy": accuracy, "speed_h_gate": speed,
                                        "gpu_total_memory_bytes": int(total), "memory": memory})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
