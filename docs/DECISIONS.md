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

## Phase 2–3 decisions

**OQ16, complex64 tolerance = 1e-5.** Measured with `python -m bench.run_tolerance`: over random
circuits with n = 1..20 and depth 100 (5 seeds per n), the largest amplitude difference
between the complex64 GPU state and the complex128 oracle was 5.5e-7, and the largest norm drift
was below 4e-7 (`bench/results/tolerance_complex64_one_qubit.json`, `..._all_gates.json`).
1e-5 leaves an ~18x margin. Single-precision rounding is ~6e-8 per operation, so the measured
error is a few rounding steps, not something growing with depth.

**OQ7, sampling = two-pass chunked.** `sample()` splits the state into 1024-amplitude chunks:
pass 1 writes one probability total per chunk (accumulated in double), the host picks a chunk per
shot from the cumulative sums, and pass 2 walks inside the chosen chunk. Extra memory is
2^n / 1024 doubles. The simple alternative (`sample_cumsum`, a float64 cumulative sum over all
amplitudes) needs 8 bytes per amplitude, which is more than the complex64 state itself (and at
30 qubits would be an extra 8 GiB). Both pass the same chi-square test against the exact
probabilities (200,000 shots, n up to 14). Their speeds have not been compared.

**Reductions accumulate in double.** Expectation values, norms and sampling chunk totals sum in
double precision even for a complex64 state. A float32 running sum over 2^28 terms would lose
digits.

**Norm uses the expectation kernel.** `amplitude_norm()` runs the Pauli kernel with empty masks
(<psi|psi>), so it needs no full-size temporary array.
