# Environment

What the project was built and measured on, and how to recreate it. Every benchmark JSON in
`bench/results/` also embeds its own environment record (`python -m bench.run_env`).

## Local development machine (measured 2026-11-19)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, Linux 6.5, x86_64 |
| CPU / RAM | 16 hardware threads, 7 GB RAM |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB, compute capability 8.6 (Ampere) |
| Driver | 535.309.01 (supports CUDA up to 12.2) |
| CUDA runtime used by CuPy | 12.9 (from pip wheels, `cupy-cuda12x[ctk]`) |
| NVRTC | 12.9 |
| nvcc | not installed (not needed: CuPy compiles kernels with NVRTC) |
| Python | 3.14.8 (installed with `uv`, in a project `.venv`) |
| numpy / scipy | 2.5.5 / 1.18.1 |
| cupy-cuda12x | 14.2.0 |
| qiskit / qiskit-aer | 2.6.2 / 0.18.4 |

Source: `bench/results/env_nvidia-geforce-rtx-3050-laptop-gpu.json`.

### Findings from the Phase 0 spike
- **CuPy RawModule works with an older driver.** The pip CUDA 12.9 runtime and NVRTC 12.9 run on driver 535
  (CUDA 12.2). Kernels compile straight to machine code for sm_86, so the driver never has to
  JIT newer PTX. The copy kernel test passes in complex64 and complex128.
- **Plain `cupy-cuda12x` is not enough.** Without the `[ctk]` extra, compiling fails with
  "Failed to find CUDA headers". The extra installs the CUDA headers and NVRTC as pip wheels
  (the venv grew from 0.7 GB to 3.2 GB).
- **qiskit-aer-gpu 0.18.4 works on this GPU** (Bell state with `device="GPU"` gives the
  expected statevector). It must live in its **own** environment: it replaces `qiskit-aer`, and
  installing it together with cuQuantum failed to resolve.
- **cuquantum-python-cu12 26.11.0 (cuStateVec 1.15.1) works on this GPU** in its own
  environment (H on qubit 0 gives the expected amplitudes).
- In a sandboxed shell the GPU reports `cudaErrorDevicesUnavailable`. GPU commands must run in a
  normal shell.
- On 2026-11-19 the driver got stuck in a faulty clock state ("System is not in ready state",
  single-digit GB/s). A reboot fixed it; all benchmarks were re-run on 2026-11-27 with normal
  clocks (copy bandwidth 168 GB/s). `bench/run_env.py` records the live clock state in every
  result, so a repeat of this would be visible.

### Memory limits that follow from 4 GB
A complex64 state needs 8 bytes × 2ⁿ: 26 qubits = 512 MiB, 27 = 1 GiB, 28 = 2 GiB,
29 = 4 GiB (does not fit). So this laptop tops out at 28 qubits in complex64 and 27 in
complex128, before workspace. Larger runs (29–30 qubits) need a 16 GB GPU such as a Colab T4.

## Recreate the environment

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev,compare]" "cupy-cuda12x[ctk]"
```

Any Python from 3.12 up works; 3.14 is what the local machine uses. CPU-only machines can skip
the CuPy line; GPU tests are then skipped automatically.

Optional comparison environments (each in its own venv):

```bash
uv venv --python 3.14 .venv-aergpu && uv pip install --python .venv-aergpu/bin/python qiskit qiskit-aer-gpu
uv venv --python 3.14 .venv-cuq && uv pip install --python .venv-cuq/bin/python cuquantum-python-cu12 "cupy-cuda12x[ctk]"
```

Offline Hamiltonian generation only (see `scripts/gen_h2_hamiltonians.py`; used pyscf 2.15.1 and
openfermion 1.8.1 on Python 3.14):

```bash
uv venv --python 3.14 .venv-chem && uv pip install --python .venv-chem/bin/python pyscf openfermion
```

## Cloud GPUs (Kaggle T4, Colab)
`scripts/gpu_cloud_run.sh` runs everything on a cloud GPU. On Kaggle it is launched with
`python scripts/kaggle_run.py push` (private script, T4 accelerator; needs the Kaggle CLI and your
own `~/.kaggle/` credentials, which the script never reads), then `status` and `fetch`. On Colab,
`notebooks/colab_run.ipynb` runs the same script. As of November 2026 Colab ships CUDA 13 with
`cupy-cuda13x` preinstalled and Python 3.13, so the script uses the preinstalled CuPy when it
can compile our kernels and installs the matching `cupy-cuda<major>x[ctk]` otherwise. Cloud
results have not been recorded yet.
