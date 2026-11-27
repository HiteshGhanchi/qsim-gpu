"""CPU-only check of GPU results: stored GPU outputs vs the NumPy oracle (W7).

CI has no GPU, so the GPU's outputs for fixed circuits are stored in tests/data/goldens/ by
scripts/make_goldens.py. This test recomputes every circuit on the CPU and compares. It also
fails when a kernel file or the CUDA backend changed after the goldens were made, because the
stored outputs would then no longer describe the current code.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qsimgpu import views
from qsimgpu.backends import NumpyBackend
from tests.goldens import OUT_DIR, kernel_hashes
from tests.tolerances import ATOL_COMPLEX64, ATOL_COMPLEX128

META = json.loads((OUT_DIR / "metadata.json").read_text())
STATES = np.load(OUT_DIR / "goldens.npz")
TOL = {"complex64": ATOL_COMPLEX64, "complex128": ATOL_COMPLEX128}


def test_goldens_match_the_current_gpu_code() -> None:
    current = kernel_hashes()
    stale = sorted({f for f in current if META["kernel_hashes"].get(f) != current[f]}
                   | {f for f in META["kernel_hashes"] if f not in current})
    assert not stale, (
        f"GPU code changed since the golden files were made ({', '.join(stale)}). "
        "Regenerate them on a machine with a GPU: python scripts/make_goldens.py "
        "(then run pytest -m gpu there too) and commit tests/data/goldens/.")


@pytest.mark.parametrize("name", sorted(META["cases"]))
def test_stored_gpu_output_matches_the_oracle(name: str) -> None:
    case = META["cases"][name]
    c = views.circuit_from_dict(case["circuit"])
    b = NumpyBackend(c.n_qubits)
    b.run(c)
    expected = b.statevector()
    for out in case["outputs"]:
        got = STATES[out["key"]].astype(np.complex128)
        np.testing.assert_allclose(got, expected, atol=TOL[out["dtype"]], err_msg=out["key"])


def test_circuit_json_round_trip() -> None:
    for case in META["cases"].values():
        c = views.circuit_from_dict(case["circuit"])
        assert views.circuit_from_dict(views.circuit_to_dict(c)).gates == c.gates
