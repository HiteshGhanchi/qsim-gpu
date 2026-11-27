"""B8: how many qubits fit? Grow n until the GPU runs out of memory.

At each n: allocate the state, build a GHZ state (H + a CX chain), check the norm, sample
1000 shots (all must be 00..0 or 11..1), and time it. Stops at the first out-of-memory error.

    python -m bench.run_b8_scale --dtype complex64 --start 24
"""

from __future__ import annotations

import argparse
import time

import cupy as cp
import numpy as np

from bench.run_env import save_result
from qsimgpu.backends.cuda_backend import CudaBackend
from qsimgpu.circuit import Circuit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    ap.add_argument("--start", type=int, default=24)
    ap.add_argument("--stop", type=int, default=34)
    args = ap.parse_args()

    rows = []
    for n in range(args.start, args.stop + 1):
        state_bytes = (1 << n) * np.dtype(args.dtype).itemsize
        try:
            t0 = time.perf_counter()
            b = CudaBackend(n, np.dtype(args.dtype))
            c = Circuit(n).h(0)
            for q in range(n - 1):
                c.cx(q, q + 1)
            b.run(c)
            b.synchronize()
            seconds = time.perf_counter() - t0
            norm = b.amplitude_norm()
            counts = b.sample(1000, seed=0)
            ok = set(counts) <= {0, (1 << n) - 1} and abs(norm - 1) < 1e-4
            rows.append({"n": n, "state_bytes": state_bytes, "ok": ok, "norm": norm,
                         "ghz_seconds": seconds, "sampled_patterns": sorted(counts)})
            print(f"n={n}: {state_bytes / 2**30:6.2f} GiB  GHZ in {seconds:7.2f} s  "
                  f"norm={norm:.6f}  correct={ok}")
            del b
            cp.get_default_memory_pool().free_all_blocks()
        except cp.cuda.memory.OutOfMemoryError as exc:
            rows.append({"n": n, "state_bytes": state_bytes, "ok": False, "oom": str(exc)[:200]})
            print(f"n={n}: {state_bytes / 2**30:6.2f} GiB  out of memory -> stop")
            break
    max_ok = max((r["n"] for r in rows if r.get("ok")), default=None)
    path = save_result(f"b8_scale_{args.dtype}", {"dtype": args.dtype, "rows": rows,
                                                  "max_qubits": max_ok})
    print(f"max qubits ({args.dtype}) = {max_ok}; saved {path}")


if __name__ == "__main__":
    main()
