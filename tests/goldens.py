"""Shared helpers for the golden files (see scripts/make_goldens.py)."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "data" / "goldens"

# Everything that decides what the GPU computes: the kernels and the code that launches them.
WATCHED = sorted([*(ROOT / "qsimgpu" / "kernels").glob("*.cu"),
                  *(ROOT / "qsimgpu" / "kernels").glob("*.cuh"),
                  ROOT / "qsimgpu" / "backends" / "cuda_backend.py"])


def kernel_hashes() -> dict[str, str]:
    """SHA-256 of every watched file, keyed by its path relative to the repository root."""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in WATCHED}
