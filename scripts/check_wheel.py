"""Check a built wheel: kernel files and py.typed are inside, and it installs and runs on its own.

    python -m build && python scripts/check_wheel.py dist/*.whl

Creates a fresh virtual environment, installs only the wheel (which pulls in NumPy), and runs a
Bell circuit on the NumPy backend from outside the source tree.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

REQUIRED = ["qsimgpu/py.typed", "qsimgpu/kernels/gates.cu", "qsimgpu/kernels/measure.cu",
            "qsimgpu/kernels/expect.cu", "qsimgpu/kernels/common.cuh"]
SMOKE = """
import qsimgpu, numpy as np
c = qsimgpu.Circuit(2).h(0).cx(0, 1)
b = qsimgpu.NumpyBackend(2)
b.run(c)
assert np.allclose(b.statevector(), [2**-0.5, 0, 0, 2**-0.5]), b.statevector()
print("qsimgpu", qsimgpu.__version__, "installed from wheel: Bell state OK")
"""


def main() -> None:
    wheels = [Path(p) for p in sys.argv[1:]]
    if len(wheels) != 1 or wheels[0].suffix != ".whl":
        sys.exit("usage: check_wheel.py <one .whl file>")
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    missing = [r for r in REQUIRED if r not in names]
    if missing:
        sys.exit(f"wheel is missing: {missing}")
    print(f"{wheels[0].name}: {len(names)} files, kernels and py.typed present")
    with tempfile.TemporaryDirectory() as tmp:
        venv.create(tmp, with_pip=True)
        py = Path(tmp) / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        subprocess.run([py, "-m", "pip", "install", "-q", str(wheels[0].resolve())], check=True)
        subprocess.run([py, "-c", SMOKE], check=True, cwd=tmp)  # cwd outside the source tree


if __name__ == "__main__":
    main()
