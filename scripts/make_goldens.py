"""Generate golden files: GPU outputs of fixed circuits, checked later on CPU-only CI.

Run on a machine with an NVIDIA GPU whenever a kernel or the CUDA backend changes:
    python scripts/make_goldens.py

Writes tests/data/goldens/goldens.npz (final statevectors) and metadata.json (the circuits as
JSON, the environment, and a SHA-256 of every file that decides what the GPU computes).
tests/test_goldens.py then recomputes every circuit with the NumPy oracle on CPU and compares,
and fails if those files changed since the goldens were made.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.run_env import environment  # noqa: E402
from qsimgpu import views  # noqa: E402
from qsimgpu.algorithms import grover, qft  # noqa: E402
from qsimgpu.backends.cuda_backend import CudaBackend  # noqa: E402
from qsimgpu.circuit import Circuit  # noqa: E402
from qsimgpu.passes.fusion import fuse  # noqa: E402
from tests.circuits import random_circuit  # noqa: E402
from tests.goldens import OUT_DIR, kernel_hashes  # noqa: E402

# Cases also run through the general (non-specialised) kernels.
GENERAL_TOO = {"random_n8_s0", "family_diagonal_n8", "family_permutation_n8", "family_general_n8"}


def family(n: int, kind: str) -> Circuit:
    """One circuit per gate family, starting from a non-trivial state."""
    c = Circuit(n)
    for q in range(n):
        c.h(q).ry(0.1 * (q + 1), q)
    if kind == "diagonal":
        c.z(0).s(1).sdg(2).t(3).tdg(4).p(0.7, 5).rz(-1.3, 6).cz(0, 7).cp(0.4, 2, 5)
        c.mcz([0, 1, 2], 3).add("rz", [4], [1, 6], [0.9])
    elif kind == "permutation":
        c.x(0).cx(1, 2).ccx(0, 3, 4).swap(5, 6).add("swap", [1, 7], [2]).mcx([0, 2, 4], 6)
    else:
        c.y(0).rx(0.3, 1).ry(-0.8, 2).u(0.2, 0.5, -0.4, 3).sx(4).sxdg(5).h(6)
        c.add("rx", [7], [0], [1.1]).add("u", [1], [2, 3], [0.3, 0.2, 0.1])
        q, _ = np.linalg.qr(np.arange(16).reshape(4, 4) + 1j * np.eye(4))
        c.unitary(q, [6, 2])
    return c


def cases() -> dict[str, Circuit]:
    out: dict[str, Circuit] = {}
    for n in (2, 4, 6, 8, 10, 12):
        for seed in (0, 1):
            out[f"random_n{n}_s{seed}"] = random_circuit(n, 80, seed=9000 + 10 * n + seed)
    for n in (4, 8, 12):
        out[f"qft_n{n}"] = qft.qft(n)
    out["grover_n6"] = grover.grover(6, 37)
    ghz = Circuit(10).h(0)
    for q in range(9):
        ghz.cx(q, q + 1)
    out["ghz_n10"] = ghz
    for kind in ("diagonal", "permutation", "general"):
        out[f"family_{kind}_n8"] = family(8, kind)
    for m in (2, 3, 4, 5):  # exercises the apply_k2 .. apply_k5 kernels
        out[f"fused_m{m}_n10"] = fuse(out["random_n10_s0"], m)
    return out


def main() -> None:
    states: dict[str, np.ndarray] = {}
    meta_cases: dict[str, Any] = {}
    for name, c in cases().items():
        variants = [("complex64", True), ("complex128", True)]
        if name in GENERAL_TOO:
            variants.append(("complex64", False))
        outputs = []
        for dtype, specialized in variants:
            b = CudaBackend(c.n_qubits, np.dtype(dtype), specialized=specialized)
            b.run(c)
            key = f"{name}__{dtype}__{'specialized' if specialized else 'general'}"
            states[key] = b.statevector()
            outputs.append({"key": key, "dtype": dtype, "specialized": specialized})
        meta_cases[name] = {"circuit": views.circuit_to_dict(c), "outputs": outputs}
        print(f"{name:24s} n={c.n_qubits:2d} gates={len(c):4d} variants={len(variants)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "goldens.npz", **states)
    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "generator": "scripts/make_goldens.py", "kernel_hashes": kernel_hashes(),
        "environment": environment(), "cases": meta_cases}, indent=1) + "\n")
    size = sum(p.stat().st_size for p in OUT_DIR.glob("*"))
    print(f"wrote {len(states)} states, {size / 1e6:.2f} MB in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
