# qsim-gpu

A from-scratch GPU statevector quantum circuit simulator, written in Python with CUDA C kernels.

Work in progress. The README will describe results once they are measured.

## Quick start

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev,compare]" "cupy-cuda12x[ctk]"
.venv/bin/pytest
```

See `docs/ENVIRONMENT.md` for details.
