"""W2: cost of step-by-step execution, and how closely GPU views match the NumPy oracle.

    python -m bench.run_step_overhead

Measures, for both backends:
- run(circuit) vs run(circuit, on_step=<does nothing>): the hook's own overhead;
- step_trace on a 20-gate, 10-qubit circuit (the W2 acceptance case);
- the largest difference between NumPy and CUDA traces / marginals over random circuits,
  which sets VIEW_ATOL_COMPLEX64 in tests/tolerances.py.
Wall-clock medians over repeated runs (the GPU is synchronised by reading results back).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from bench.run_env import save_result
from qsimgpu import views
from qsimgpu.backends import NumpyBackend
from qsimgpu.backends.cuda_backend import CudaBackend
from tests.circuits import random_circuit, random_state


def median_ms(fn: Callable[[], Any], repeats: int = 7) -> float:
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    makers: dict[str, Callable[[int], Any]] = {"numpy": NumpyBackend, "cuda": CudaBackend}
    overhead = []
    for n, depth in ((10, 20), (16, 200), (20, 200)):
        c = random_circuit(n, depth, seed=7)
        for name, make in makers.items():
            b = make(n)

            def plain(b: Any = b, c: Any = c) -> None:
                b.reset()
                b.run(c)
                b.amplitude_norm()  # forces the GPU to finish

            def hooked(b: Any = b, c: Any = c) -> None:
                b.reset()
                b.run(c, on_step=lambda i, g, bk: None)
                b.amplitude_norm()

            p, h = median_ms(plain), median_ms(hooked)
            overhead.append({"backend": name, "n": n, "gates": depth, "plain_ms": p,
                             "with_on_step_ms": h, "overhead_us_per_gate": (h - p) * 1e3 / depth})
            print(f"{name:6s} n={n:2d} gates={depth:3d}  plain {p:8.3f} ms  "
                  f"on_step {h:8.3f} ms  ({(h - p) * 1e3 / depth:+.2f} us/gate)")

    c = random_circuit(10, 20, seed=5)
    trace_ms = {name: median_ms(lambda make=make: views.step_trace(make(10), c))
                for name, make in makers.items()}
    print("step_trace 10 qubits, 20 gates:", {k: round(v, 2) for k, v in trace_ms.items()})

    trace_diff = max(views.trace_max_difference(
        views.step_trace(NumpyBackend(10), random_circuit(10, 20, seed=s)),
        views.step_trace(CudaBackend(10), random_circuit(10, 20, seed=s))) for s in range(20))
    rng = np.random.default_rng(0)
    marg_diff = 0.0
    for s in range(10):
        psi = random_state(20, s)
        a, b = NumpyBackend(20), CudaBackend(20)
        a.set_state(psi)
        b.set_state(psi)
        qs = [int(q) for q in rng.permutation(20)[: int(rng.integers(1, 13))]]
        marg_diff = max(marg_diff, float(np.max(np.abs(a.marginal_probabilities(qs)
                                                       - b.marginal_probabilities(qs)))))
    print(f"max |NumPy - CUDA|: traces {trace_diff:.2e} (20 circuits, 10 qubits), "
          f"marginals {marg_diff:.2e} (10 states, 20 qubits, 1-12 qubits)")
    path = save_result("w2_steps", {"on_step_overhead": overhead, "step_trace_ms": trace_ms,
                                    "trace_max_abs_diff": trace_diff,
                                    "marginal_max_abs_diff_n20": marg_diff})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
