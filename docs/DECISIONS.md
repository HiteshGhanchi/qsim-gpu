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

## Phase 4 decisions

**OQ19, Grover iteration count = floor.** `optimal_iterations()` evaluates both floor and round
of (pi/4)·sqrt(2^n) and keeps the one with the higher theoretical success probability
sin²((2k+1)θ). For n = 2..20 the two differ only at n = 2, 7, 8, 9, 11, 14, 19, and floor wins
every time (for example n = 2: floor = 1 round gives 1.0, round = 2 rounds gives 0.25; n = 8:
0.999947 vs 0.986186). The function still compares both rather than hard-coding floor.

**Shor-15 multiplication uses a^4 = 1 (mod 15).** The controlled multiplication by a^(2^j) is
applied as (2^j mod 4) repetitions of "multiply by a", since every valid a has a^4 ≡ 1 mod 15. A
test checks every circuit against y → a^p·y mod 15 for all y = 1..14 and p = 1..4 before the
algorithm uses it.

**Continued fractions may return a divisor of r.** A measured value such as 128/256 = 1/2 gives
r = 2 for a = 7, which fails the check a^r ≡ 1 (mod 15). The post-processing then tries 2r and 4r,
which recovers r = 4. That's the standard fix when s and r share a factor.

## Phase 5 decisions and results (VQE for H2)

**Hamiltonians are generated offline and committed.** `scripts/gen_h2_hamiltonians.py` (PySCF 2.15.1 +
OpenFermion 1.8.1) writes `data/h2_hamiltonians.json` for 45 bond lengths, 0.30–2.50 Å. Before writing,
it checks that the 2-qubit reduced Hamiltonian's lowest eigenvalue equals the PySCF FCI energy within
1e-8 at every point. Runtime code reads the JSON with NumPy only.

**Exact reference = 2-electron sector.** The 4-qubit Jordan–Wigner Hamiltonian also describes 0, 1, 3 and
4 electrons. The reference energy is the lowest eigenvalue among basis states with exactly two 1s, which
is the sector the UCCSD ansatz stays in. A test checks it equals the FCI energy (1e-8) at all 45 points.

**UCCSD generators are built numerically.** Fermion operators are written as Jordan–Wigner matrices
(16×16), and each excitation T − T† is split into Pauli strings with c_P = Tr(P·G)/16. The double
excitation becomes 8 Pauli strings and each single becomes 2. All strings inside one generator commute, so
one Trotter step reproduces exp(θ·G) exactly (tested against `scipy.linalg.expm`).

**exp(iφP) circuit.** Basis change (H for X, Rx(π/2) for Y), a CX ladder collecting the parity on the
lowest active qubit, Rz(−2φ) there, then the undo. It's tested against `expm` on random states and Pauli strings.

**Results** (stored in `bench/results/vqe_h2_*.json`, optimiser L-BFGS-B with finite differences,
warm start from the previous bond length):

| Run | Worst \|VQE − exact\| over 45 points | Minimum |
|---|---|---|
| 2-qubit reduced, NumPy | 3.0e-11 Ha | R = 0.75 Å, E = −1.137117 Ha |
| 4-qubit UCCSD, NumPy (complex128) | 1.4e-10 Ha | R = 0.75 Å, E = −1.137117 Ha |
| 4-qubit UCCSD, GPU (complex64) | 6.3e-5 Ha | R = 0.75 Å, E = −1.137115 Ha |

All are well inside chemical accuracy (1.6e-3 Ha). The two single-excitation angles stay below
4.5e-6 at every point (expected ~0 by symmetry, and now measured rather than assumed). On a 4-qubit state the GPU run
was slower than NumPy (22.6 s vs 10.2 s wall time): each kernel launch costs more than the
16-amplitude update it performs. The GPU only pays off for large states.
