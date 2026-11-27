#!/usr/bin/env bash
# Run the full test suite and every benchmark on a cloud GPU (Kaggle or Colab), then zip the results.
# Run from the repository root:  bash scripts/gpu_cloud_run.sh
# Optional parts (Aer GPU, cuStateVec) may fail on some images; that is logged, not fatal.
set -uo pipefail

log() { echo; echo "=== $*"; }
log "machine"; nvidia-smi; python --version
CUDA_MAJOR=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+" || echo 12)
echo "driver supports CUDA ${CUDA_MAJOR}"

log "CuPy"
# Our kernels include <cupy/complex.cuh>, which needs the CUDA headers at compile time.
if python - <<'PY'
import cupy as cp
cp.RawKernel(r'#include <cupy/complex.cuh>' '\nextern "C" __global__ void k(complex<float>* x){ x[0] *= 2; }',
             "k", options=("-std=c++17",)).compile()
PY
then
  echo "preinstalled CuPy works"
else
  echo "installing cupy-cuda${CUDA_MAJOR}x[ctk]"
  pip uninstall -y -q cupy-cuda11x cupy-cuda12x cupy-cuda13x || true
  pip install -q "cupy-cuda${CUDA_MAJOR}x[ctk]"
fi

log "install qsim-gpu"
pip install -q -e ".[dev,compare]" || exit 1

log "tests (CPU + GPU)"
python -m pytest -q -rs 2>&1 | tee test_log.txt

log "benchmarks (16 GB settings)"
python -m bench.run_all --big 2>&1 | tee bench_log.txt

log "comparison engines in their own environments"
pip install -q uv
uv venv -q /tmp/aergpu && uv pip install -q --python /tmp/aergpu/bin/python qiskit qiskit-aer-gpu \
  && for e in aer-gpu aer-gpu-nofusion; do
       PYTHONPATH=. /tmp/aergpu/bin/python -m bench.run_b7_compare --engine "$e" 2>&1 | tee -a bench_log.txt
     done || echo "Aer GPU comparison failed" | tee -a bench_log.txt
uv venv -q /tmp/cuq && uv pip install -q --python /tmp/cuq/bin/python \
  "cuquantum-python-cu${CUDA_MAJOR}" "cupy-cuda${CUDA_MAJOR}x[ctk]" \
  && PYTHONPATH=. /tmp/cuq/bin/python -m bench.run_b7_compare --engine custatevec 2>&1 | tee -a bench_log.txt \
  || echo "cuStateVec comparison failed" | tee -a bench_log.txt

log "package results"
python -m bench.report --print > report_preview.md 2>&1 || true
zip -qr results.zip bench/results test_log.txt bench_log.txt report_preview.md
echo "results.zip written"
