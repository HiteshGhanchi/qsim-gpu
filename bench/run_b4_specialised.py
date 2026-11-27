"""B4: do the specialised kernels beat the general ones?

For each gate, times the general path (specialized=False) and the specialised path on the same
state, back to back in one process, next to a copy of the same state. Reports time per gate,
speedup and the bytes each path actually needs to move.

    python -m bench.run_b4_specialised --qubits 24
"""

from __future__ import annotations

import argparse

import cupy as cp
import numpy as np

from bench.run_env import save_result, time_gpu
from qsimgpu.backends.cuda_backend import BLOCK, MAX_BLOCKS, CudaBackend, _module
from qsimgpu.circuit import Gate


def cases(n: int) -> list[tuple[str, Gate, float]]:
    """(label, gate, fraction of the state the specialised kernel must read+write)."""
    hi = n - 1
    return [
        ("h (general both ways)", Gate("h", (hi,)), 1.0),
        ("x", Gate("x", (hi,)), 1.0),
        ("z", Gate("z", (hi,)), 0.5),
        ("t", Gate("t", (hi,)), 0.5),
        ("rz", Gate("rz", (hi,), params=(0.3,)), 1.0),
        ("cx", Gate("cx", (hi,), (hi - 1,)), 0.5),
        ("cz", Gate("cz", (hi,), (hi - 1,)), 0.25),
        ("cp", Gate("cp", (hi,), (hi - 1,), (0.7,)), 0.25),
        ("ccx", Gate("ccx", (hi,), (hi - 1, hi - 2)), 0.25),
        ("swap", Gate("swap", (hi, hi - 1)), 0.5),
        ("mcz (all other qubits control)", Gate("z", (hi,), tuple(range(hi))), 2.0**-n),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=24)
    ap.add_argument("--per-timing", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    n = args.qubits
    size = 1 << n
    b = CudaBackend(n, np.complex64)
    copy = _module("gates.cu", "complex64").get_function("copy_state")
    dst = cp.empty_like(b.state)
    grid = (max(1, min(MAX_BLOCKS, (size + BLOCK - 1) // BLOCK)),)
    t_copy = time_gpu(lambda: [copy(grid, (BLOCK,), (b.state, dst, np.int64(size)))
                               for _ in range(args.per_timing)], repeats=args.repeats)
    copy_ms = t_copy["median_ms"] / args.per_timing
    print(f"copy of the state: {copy_ms:.3f} ms")
    rows = []
    for label, g, fraction in cases(n):
        result = {}
        for mode in (False, True):
            b.specialized = mode
            t = time_gpu(lambda g=g: [b.apply(g) for _ in range(args.per_timing)],
                         repeats=args.repeats)
            result["specialized" if mode else "general"] = t | {
                "ms_per_gate": t["median_ms"] / args.per_timing}
        gen, spec = result["general"]["ms_per_gate"], result["specialized"]["ms_per_gate"]
        rows.append({"case": label, "gate": g.name, "controls": len(g.controls),
                     "touched_fraction_specialised": fraction, **result,
                     "speedup": gen / spec, "specialised_vs_copy": spec / copy_ms})
        print(f"{label:32s} general {gen:9.3f} ms  specialised {spec:9.3f} ms  "
              f"x{gen / spec:5.2f}  (copy = {copy_ms:.3f} ms)")
    path = save_result("b4_specialised", {"n": n, "dtype": "complex64",
                                          "copy_ms": copy_ms, "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
