"""Compare the Pyodide run of web/pyodide-check/workload.py with the same workload in CPython.

    python scripts/compare_pyodide.py web/pyodide-check/pyodide_out.json

Every number must agree within ATOL_COMPLEX128: both runs use the double-precision NumPy backend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "pyodide-check"))

import workload  # noqa: E402

from qsimgpu import views  # noqa: E402
from tests.tolerances import ATOL_COMPLEX128  # noqa: E402


def max_diff(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    theirs = json.loads(Path(sys.argv[1]).read_text())
    ours = json.loads(workload.main_json())
    r, o = theirs["results"], ours["results"]
    diffs = {"bell": max_diff(r["bell"], o["bell"]),
             "ghz_probabilities": max_diff(r["ghz_probabilities"], o["ghz_probabilities"]),
             "trace": views.trace_max_difference(r["trace"], o["trace"])}
    print("Pyodide:", theirs["info"])
    print("CPython:", ours["info"])
    print("max |difference|:", {k: f"{v:.1e}" for k, v in diffs.items()})
    bad = {k: v for k, v in diffs.items() if v > ATOL_COMPLEX128}
    if bad:
        sys.exit(f"Pyodide and CPython disagree beyond {ATOL_COMPLEX128}: {bad}")
    print(f"OK: all within ATOL_COMPLEX128 = {ATOL_COMPLEX128}")


if __name__ == "__main__":
    main()
