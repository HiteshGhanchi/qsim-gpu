"""Run the H2 VQE scan and store the curve (VQE vs exact vs Hartree-Fock).

    python -m bench.run_vqe_h2 --backend numpy --stage 4q
    python -m bench.run_vqe_h2 --backend cuda --stage 4q
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

from bench.run_env import save_result
from qsimgpu.algorithms import vqe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "cuda"], default="numpy")
    ap.add_argument("--stage", choices=["2q", "4q"], default="4q")
    args = ap.parse_args()

    if args.backend == "numpy":
        from qsimgpu.backends import NumpyBackend as make
        fd_step = 1e-5
    else:
        from qsimgpu.backends.cuda_backend import CudaBackend as make  # type: ignore[assignment]
        fd_step = 1e-4  # complex64 energies carry ~1e-7 noise; a larger step keeps gradients clean

    t0 = time.perf_counter()
    scan = vqe.run_scan(make, args.stage, fd_step=fd_step)  # type: ignore[arg-type]
    seconds = time.perf_counter() - t0

    rows = [asdict(p) | {"error": p.error} for p in scan.points]
    best = min(scan.points, key=lambda p: p.energy)
    worst = max(scan.points, key=lambda p: abs(p.error))
    print(f"{'R':>5} {'VQE':>11} {'exact':>11} {'HF':>11} {'error':>10}")
    for p in scan.points:
        print(f"{p.bond_length:5.2f} {p.energy:11.6f} {p.exact:11.6f} {p.hf:11.6f} {p.error:10.2e}")
    print(f"minimum at R={best.bond_length:.2f} A, E={best.energy:.6f} Ha; "
          f"worst |error| {abs(worst.error):.2e} Ha at R={worst.bond_length:.2f}; {seconds:.1f} s")
    path = save_result(f"vqe_h2_{args.stage}_{args.backend}", {
        "stage": args.stage, "backend": args.backend, "fd_step": fd_step,
        "wall_seconds": seconds, "max_abs_error": abs(worst.error),
        "minimum": {"bond_length": best.bond_length, "energy": best.energy}, "points": rows,
    })
    print(f"saved {path}")


if __name__ == "__main__":
    main()
