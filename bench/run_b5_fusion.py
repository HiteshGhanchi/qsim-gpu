"""B5: does gate fusion help, and what fused width m is best?

For QFT and a depth-100 random circuit, compares the unfused circuit (general and specialised
kernels) with fused versions for m = 1..5. Reports the number of passes over the state
(= gates after fusion), GPU time for the whole circuit, and how long the fusion pass took on
the host (not included in the GPU time).

    python -m bench.run_b5_fusion --qubits 22
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import numpy as np

from bench.run_env import save_result
from qsimgpu.algorithms.qft import qft
from qsimgpu.backends.cuda_backend import CudaBackend
from qsimgpu.circuit import Circuit
from qsimgpu.passes.fusion import fuse
from tests.circuits import random_circuit


def time_circuit(b: CudaBackend, c: Circuit, repeats: int) -> dict[str, Any]:
    b.reset()
    b.run(c)
    b.synchronize()  # warm-up
    ts = []
    for _ in range(repeats):
        b.reset()
        b.synchronize()
        t0 = time.perf_counter()
        b.run(c)
        b.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts),
            "all_ms": ts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=22)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    n = args.qubits
    b = CudaBackend(n, np.complex64)
    rows = []
    for name, c in {"qft": qft(n), "random_d100": random_circuit(n, 100, seed=2024)}.items():
        variants: list[tuple[str, Circuit, bool, float]] = [
            ("unfused, general kernels", c, False, 0.0),
            ("unfused, specialised kernels", c, True, 0.0),
        ]
        for m in range(1, 6):
            t0 = time.perf_counter()
            f = fuse(c, m)
            variants.append((f"fused m={m}", f, True, (time.perf_counter() - t0) * 1e3))
        base_ms = None
        for label, circ, spec, fuse_ms in variants:
            b.specialized = spec
            t = time_circuit(b, circ, args.repeats)
            base_ms = base_ms or t["median_ms"]
            rows.append({"circuit": name, "variant": label, "passes": len(circ),
                         "fusion_host_ms": fuse_ms, "timing": t,
                         "speedup_vs_unfused_general": base_ms / t["median_ms"]})
            print(f"{name:12s} {label:30s} passes={len(circ):4d}  {t['median_ms']:9.1f} ms  "
                  f"x{base_ms / t['median_ms']:5.2f}  (fusion took {fuse_ms:7.1f} ms on host)")
    path = save_result("b5_fusion", {"n": n, "dtype": "complex64", "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
