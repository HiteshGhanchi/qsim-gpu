"""Packaging: version, public API, and that the core import needs only NumPy."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

import qsimgpu

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert qsimgpu.__version__ == meta["project"]["version"]
    assert meta["project"]["dependencies"] == ["numpy>=2.0"]


def test_public_api_is_exported() -> None:
    for name in qsimgpu.__all__:
        assert getattr(qsimgpu, name) is not None


def test_core_import_does_not_load_cupy_or_scipy() -> None:
    code = ("import sys, qsimgpu; "
            "c = qsimgpu.Circuit(2).h(0).cx(0, 1); b = qsimgpu.NumpyBackend(2); b.run(c); "
            "print(sorted(m for m in ('cupy', 'scipy', 'qiskit') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_typed_marker_and_kernels_are_package_data() -> None:
    assert files("qsimgpu").joinpath("py.typed").is_file()
    names = {p.name for p in files("qsimgpu.kernels").iterdir()}
    assert {"gates.cu", "measure.cu", "expect.cu", "common.cuh"} <= names
