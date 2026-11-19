# Decisions log

Approved change requests and resolved [OPEN] items, newest last.

## CR-001: add mypy (approved 2026-11-19)

```
CHANGE REQUEST
What I want to change: add mypy (dev-only) and run it in strict mode on qsim/
Why: owner wants TypeScript-style type safety; type hints alone are never checked
What it affects: pyproject.toml (dev deps + config), one extra check command; no runtime change
Cost/risk: small; NumPy/CuPy types are sometimes loose, so a few `# type: ignore` lines may be needed
Alternative that keeps the current scope: write hints but never check them
Owner decision: approved   Date: 2026-11-19
```

mypy added to the dev/test dependency list.

## Phase 0 decisions (2026-11-19)

| ID | Question | Decision | Evidence |
|---|---|---|---|
| OQ1 | Which GPU? | Develop and run GPU tests on the local RTX 3050 Laptop (4 GB). Runs above 28 qubits and headline numbers for a 16 GB card go to a Colab T4. | `nvidia-smi`; `docs/ENVIRONMENT.md` |
| OQ2 | Time budget | Build all core phases (0–8) in one pass, in phase order, committing at each milestone. Stretch items stay out. | owner |
| OQ4 | How kernels are built | CuPy `RawModule` compiling `.cu` text with NVRTC at run time. No nvcc and no C++ extension build. | copy-kernel test passes in complex64 and complex128 |
| OQ5 | VQE staging | 2-qubit reduced Hamiltonian first, then 4-qubit Jordan–Wigner UCCSD. | owner |
| OQ6 | Shor | Hard-coded circuit for N = 15. General N stays a stretch item. | owner |
| OQ8 | Do Aer-GPU and cuQuantum install? | Both install and run on the local GPU, each in its own environment (they conflict with each other and with `qiskit-aer`). | Phase 0 spike, `docs/ENVIRONMENT.md` |
| OQ11 | Repo | `qsim-gpu`, public, MIT licence. | owner |
| OQ12 | Where development runs | Native Ubuntu 24.04 on the laptop. CPU-only work runs there too. | `lsb_release` |
| OQ13 | Python | 3.14 locally (installed with `uv`); the package supports 3.12+ so Colab works. | all wheels needed exist for 3.14; local tests pass |
| OQ14 | Noise / density matrices | Out of scope. | owner |
| OQ15 | Plots | No plots or images for now. Results are reported as JSON and text tables. | owner |
| OQ17 | CI | GitHub Actions runs the CPU tests only (free runners have no GPU). GPU tests run locally and on Colab. | owner |
| OQ20 | VQE bond-length grid | 0.3 to 2.5 Å, step 0.05 Å. | owner |

Other choices made while building Phase 0:
- **Shared kernel header.** `qsim/kernels/common.cuh` holds the precision typedef target and the
  `insert_zero_bits` helper, so `gates.cu`, `measure.cu` and `expect.cu` don't repeat them.
- **One source, two precisions.** The backend prepends `typedef float real_t;` or
  `typedef double real_t;` and compiles the same `.cu` text twice. This keeps the complex64 vs
  complex128 study honest: both use identical kernel code.
- **GPU tests skip themselves** when CuPy or a usable device is missing, so `pytest` works on
  any machine.
