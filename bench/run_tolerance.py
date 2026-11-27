"""Calibrate the complex64 test tolerance: how far does the GPU state drift from the oracle?

For random circuits (n = 1..20, depth 100) compare the complex64 CUDA backend with the
complex128 NumPy oracle and record the largest |amplitude difference| and norm drift.

    python -m bench.run_tolerance            # full gate set
    python -m bench.run_tolerance --one-qubit
"""

from __future__ import annotations

import argparse

import numpy as np

from bench.run_env import save_result
from qsimgpu.backends import NumpyBackend
from qsimgpu.backends.cuda_backend import CudaBackend
from tests.circuits import random_circuit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-qubit", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=100)
    args = ap.parse_args()

    rows = []
    for n in range(1, 21):
        worst_amp = worst_norm = 0.0
        for seed in range(args.seeds):
            c = random_circuit(n, args.depth, 1000 * n + seed, only_one_qubit=args.one_qubit)
            ref = NumpyBackend(n)
            ref.run(c)
            gpu = CudaBackend(n, np.complex64)
            gpu.run(c)
            sv = gpu.statevector().astype(np.complex128)
            worst_amp = max(worst_amp, float(np.max(np.abs(sv - ref.statevector()))))
            worst_norm = max(worst_norm, abs(float(np.linalg.norm(sv)) - 1.0))
        rows.append({"n": n, "max_abs_amp_error": worst_amp, "max_norm_error": worst_norm})
        print(f"n={n:2d}  max|diff|={worst_amp:.3e}  norm drift={worst_norm:.3e}")

    overall = max(r["max_abs_amp_error"] for r in rows)
    name = "tolerance_complex64_" + ("one_qubit" if args.one_qubit else "all_gates")
    path = save_result(name, {"depth": args.depth, "seeds": args.seeds,
                              "one_qubit_only": args.one_qubit, "rows": rows,
                              "overall_max_abs_amp_error": overall,
                              "overall_max_norm_error": max(r["max_norm_error"] for r in rows)})
    print(f"overall max |diff| = {overall:.3e}; saved {path}")


if __name__ == "__main__":
    main()
