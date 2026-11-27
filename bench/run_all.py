"""Re-run every benchmark in order (use after a reboot, or on Colab).

    python -m bench.run_all            # laptop defaults (4 GB GPU)
    python -m bench.run_all --big      # 16 GB GPU (e.g. Colab T4): larger n, scale run to 30+

Each step writes its own JSON to bench/results/. Steps that need a different Python
environment (Aer GPU, cuStateVec) are printed as commands instead of run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="settings for a 16 GB GPU")
    args = ap.parse_args()
    top = "29" if args.big else "27"
    n_mid = "26" if args.big else "24"
    steps = [
        ["bench.run_env"],
        ["bench.run_tolerance"],
        ["bench.run_b0_bandwidth", "--max-qubits", top],
        ["bench.run_b1_target", "--qubits", "26", "--per-timing", "3"],
        ["bench.run_b2_scaling", "--max-qubits", "30" if args.big else "28", "--per-timing", "3"],
        ["bench.run_b3_efficiency", "--qubits", n_mid, "--generic", "--tag", "general"],
        ["bench.run_b3_efficiency", "--qubits", n_mid, "--tag", "specialised"],
        ["bench.run_b4_specialised", "--qubits", n_mid],
        ["bench.run_b5_fusion", "--qubits", "24" if args.big else "22"],
        ["bench.run_b6_precision"],
        ["bench.run_b7_compare", "--engine", "qsim"],
        ["bench.run_b7_compare", "--engine", "qsim-fused"],
        ["bench.run_b7_compare", "--engine", "aer-cpu"],
        ["bench.run_b7_compare", "--engine", "aer-cpu-nofusion"],
        ["bench.run_b8_scale", "--dtype", "complex64", "--start", "26"],
        ["bench.run_b8_scale", "--dtype", "complex128", "--start", "26"],
        ["bench.run_vqe_h2", "--backend", "cuda", "--stage", "4q"],
    ]
    for step in steps:
        print(f"\n=== python -m {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True)
    print("\nRun these in their own environments (see docs/ENVIRONMENT.md):")
    for env, engine in ((".venv-aergpu", "aer-gpu"), (".venv-aergpu", "aer-gpu-nofusion"),
                        (".venv-cuq", "custatevec")):
        print(f"  PYTHONPATH=. {env}/bin/python -m bench.run_b7_compare --engine {engine}")


if __name__ == "__main__":
    main()
