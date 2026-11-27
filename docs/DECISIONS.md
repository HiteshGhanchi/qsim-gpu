# Decisions log

Approved change requests and resolved [OPEN] items, newest last.

## CR-001: add mypy (approved 2026-11-19)

```
CHANGE REQUEST
What I want to change: add mypy (dev-only) and run it in strict mode on qsimgpu/
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
- **Shared kernel header.** `qsimgpu/kernels/common.cuh` holds the precision typedef target and the
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

## Phase 6: baseline measurements (before any optimisation)

**Caveat that affects every number below.** During these runs the laptop GPU driver reported
"System is not in ready state" for its clocks (recorded in each JSON under
`environment.gpu.state_at_run` and `warning`). DRAM copy bandwidth was only 1.5–9 GB/s and
drifted between runs, while an RTX 3050 Laptop GPU is specified far higher. So the
**absolute** numbers are not representative, and comparisons are only fair *within one run*.
These benchmarks must be re-run after a reboot (and on a Colab T4) before any headline figure
is quoted. The scripts and the analysis method don't change.

What the baseline showed, with that caveat:

- **B0, bandwidth** (`b0_bandwidth.json`). Our copy kernel and `cupy.copyto` give the same
  bandwidth at every size (within ~5%), so the copy kernel is a fair yardstick. Small buffers
  (2^16 amplitudes = 512 KiB) ran far faster than large ones in the first run: they fit in the
  2 MiB L2 cache.
- **B2, scaling** (`b2_scaling_complex64.json`). Up to 17 qubits (1 MiB state) a gate takes a flat
  ~0.009–0.010 ms: the state sits in L2 and the time is mostly launch overhead. From 18 qubits on,
  every extra qubit doubles the time (×2.00 per step up to 28 qubits). That's the
  memory-bound regime the roofline argument predicts.
- **B1, target qubit** (`b1_target_baseline.json`, n = 26). Targets 0–7 took ~700 ms per H gate and
  targets 8–25 ~135 ms, about 5× slower for low targets. But B3 at n = 24, measured minutes later, found no low-target
  penalty. Given the unstable clocks, this stays an **open observation** (OQ10) to re-measure
  on a healthy GPU, not a conclusion.
- **B3, efficiency** (`b3_efficiency_baseline.json`, n = 24, same process as its copy timing).
  The general 1-qubit kernel ran at 99.6–100% of copy bandwidth for H, Rz and X. Controlled gates
  (CX, CZ) took half the copy time, since they only touch the half of the state where the control
  is 1. SWAP through the general 4×4 kernel took 1.8× the copy time. That kernel keeps 32
  complex values per thread in a local array, which likely spills to local memory: a
  candidate for a specialised kernel.
- **B7, comparison** (`b7_compare_*.json`, complex64, QFT and a depth-100 random circuit,
  n = 10–24). Below 18 qubits, qsim and cuStateVec take about 1 ms per circuit while Aer (CPU or GPU) takes 3–13 ms,
  because Aer has a fixed per-run setup cost. From 18 qubits up, qsim and cuStateVec track
  each other closely (n = 24 QFT: 6.9 s vs 6.6 s), and Aer GPU is the fastest (3.7 s). The likely
  reason: Aer fuses consecutive gates by default, so it makes fewer passes over memory. That's
  exactly what the fusion pass in Phase 7 tests. Aer CPU (16 threads) is close to the
  degraded GPU runs, which again points to the GPU's memory clock being the bottleneck here.

## Phase 7.1: specialised kernels (B4)

Four new kernels: `apply_phase` (diag(1, φ) gates: Z, S, T, P, CZ, CP, multi-controlled Z),
`apply_diag` (Rz), `apply_x` (X, CX, CCX, multi-controlled X) and `apply_swap`. `CudaBackend(...,
specialized=False)` keeps the old general path for comparison. Both paths are tested against
explicit matrices for every gate and against each other on random circuits.

Measured at n = 24, complex64, general vs specialised in the same process (`b4_specialised.json`;
same GPU clock caveat as Phase 6, but both paths ran under identical conditions):

| Gate | General | Specialised | Speedup | Why |
|---|---|---|---|---|
| H | 33.7 ms | 33.8 ms | 1.00× | no specialised kernel (general family) |
| X | 33.8 ms | 33.7 ms | 1.00× | fewer multiplications, same bytes: memory-bound, so no gain |
| Rz | 33.8 ms | 33.3 ms | 1.01× | one read + write per amplitude either way |
| Z, T | 33.7 ms | 16.6 ms | 2.03× | amplitudes with target bit 0 are never read |
| CX | 16.9 ms | 16.8 ms | 1.01× | the general kernel already skipped control = 0 |
| CZ, CP | 16.9 ms | 8.3 ms | 2.03× | only the \|11⟩ quarter is touched |
| CCX | 8.5 ms | 8.5 ms | 1.00× | same bytes |
| SWAP | 60.6 ms | 16.9 ms | 3.57× | only \|01⟩ and \|10⟩ move, and no 4×4 matrix in local memory |
| Z with 23 controls | 0.012 ms | 0.009 ms | 1.33× | a single amplitude; launch overhead dominates |

**What this shows:** removing arithmetic did nothing (X, CX, Rz). Every speedup came from
moving fewer bytes. That's the clearest evidence that the simulator is memory-bound. The
copy kernel took 33.6 ms on the same state, so specialised Z/T/CZ/CP and SWAP now run
*faster than a full copy*: they don't need to touch the whole state.

## Phase 7.2: gate fusion (B5) and OQ9

`qsimgpu/passes/fusion.py` greedily merges consecutive gates while the union of their qubits stays
≤ m and replaces each block with one `unitary` gate (its matrix is computed by running the block on
a small NumPy simulator, one column per basis state). Blocks of one gate stay unchanged so they keep
their specialised kernel. Fused circuits are tested against the unfused ones for m = 1..5.

**First attempt: fewer passes, slower circuits** (`b5_fusion_v1_untemplated.json`, n = 22). For
the depth-100 random circuit, m = 3 cut the passes from 100 to 55, yet ran only 1.19× faster
than the general-kernel baseline, slower than the unfused specialised kernels (1.42×). At m = 5,
65 passes took 9.1 s against 1.5 s unfused. There were two causes:
1. The multi-target kernel used a runtime k and arrays sized for the maximum (32 amplitudes per
   thread). That doesn't fit in registers, so it spills to local memory, adding memory traffic
   in a memory-bound kernel.
2. A fused block always reads and writes the whole state, so it loses the savings of
   specialised kernels (a CP alone touches a quarter of the state).

**Fix: one compiled kernel per k** (`apply_k1` … `apply_k5`, a C++ template with compile-time K,
unrolled loops, matrix in shared memory). Re-measured (`b5_fusion.json`, same n and circuits,
specialised kernels for unfused gates):

| Circuit | Unfused (general) | Unfused (specialised) | m=2 | m=3 | m=4 | m=5 |
|---|---|---|---|---|---|---|
| random depth 100: passes | 100 | 100 | 79 | 55 | 42 | 31 |
| random depth 100: time | 675 ms | 512 ms | 421 ms | **325 ms** | 2306 ms | 6369 ms |
| QFT: passes | 264 | 264 | 242 | 131 | 88 | 65 |
| QFT: time | 1421 ms | 810 ms | **752 ms** | 1025 ms | 7617 ms | 15876 ms |

**Decision (OQ9): default fused width m = 3** (best for random circuits: 2.08× over the general
baseline, 1.58× over specialised kernels alone). For QFT, m = 2 is best and m = 3 already loses to
the unfused specialised run, because most QFT gates are controlled phases that the specialised
kernel handles on a quarter of the state. m ≥ 4 is much slower here. A 16×16 or 32×32 complex
multiply per amplitude group is no longer memory-bound on this GPU, and the unrolled kernel
needs more registers than are available. All of this was measured with the GPU's clocks in the
degraded state described in Phase 6, so the best m should be re-checked on a healthy GPU and
on the T4.

Fusion cost on the host: 4–130 ms per circuit (m = 2..5), paid once per circuit, not per run.

## Phase 7.3: precision study (B6)

Measured in `b6_precision.json`, random circuits of depth 100 against the complex128 NumPy oracle,
3 seeds per n:

| n | complex64 max error | complex128 max error |
|---|---|---|
| 10 | 1.1e-7 | 1.7e-16 |
| 14 | 5.3e-8 | 8.8e-17 |
| 18 | 5.7e-8 | 1.1e-16 |
| 20 | 6.2e-8 | 7.2e-17 |

Time for one H gate (same GPU state caveat as Phase 6):
complex128 / complex64 = 2.00–2.01 at n = 20, 22, 24, 26. That's exactly the ratio of bytes per
amplitude (16 vs 8). Double precision costs no more than its extra memory traffic. Arithmetic,
even on a GPU with slow FP64 units, is not the bottleneck.

Memory: one qubit fewer in complex128. On the 4 GB laptop, B8 (below) confirms 28 qubits in
complex64 and 27 in complex128.

**Decision: complex64 stays the default.** Its error (~1e-7) is about 10,000× below what any
test or algorithm here needs. VQE in complex64 stayed within 6.3e-5 Ha of exact, against a
1.6e-3 Ha requirement. And it allows one more qubit and halves the time per gate.

## Phase 8: scale limit (B8)

`b8_scale_complex64.json` / `b8_scale_complex128.json`: build a GHZ state at growing n, check the
norm and that 1000 samples only ever give 00…0 or 11…1, and stop at the first out-of-memory error.

| dtype | largest n that ran correctly | first failure |
|---|---|---|
| complex64 | 28 (2 GiB state) | 29 (4 GiB) out of memory |
| complex128 | 27 (2 GiB state) | 28 (4 GiB) out of memory |

The 29–30 qubit runs need a 16 GB GPU (T4 on Colab). The notebook includes that run, but it has
not been run yet.

## Documentation added at the owner's request

- `docs/CODEBASE_GUIDE.md`: a file-by-file guide covering what each part does, why it's built
  that way, the problems hit while building it, and likely interview questions.
- `lessons/lesson1_cuda.py` and sections 14–15 of `lessons/teaching.md`: the GPU side (CuPy
  arrays, a raw CUDA kernel, timing) and the commands used day to day.
- README without images for now (OQ15); results are described in text and tables.

## Re-measured on a healthy GPU (2026-11-27): supersedes the timing figures above

After a reboot, the GPU driver reported normal clocks again (memory 5870 MHz). Its clock-event
flags still showed software power and thermal capping during the runs; that's recorded in every
JSON. Every benchmark was re-run from a clean commit and the files in `bench/results/` were
replaced. The Phase 6–8 sections above describe the earlier runs and are kept for the record.
**Where their numbers disagree with this section, this section is correct.** Only
`b5_fusion_v1_untemplated.json` and `tolerance_complex64_one_qubit.json` remain from before.

| Benchmark | Healthy-GPU result | Change from the degraded runs |
|---|---|---|
| B0 copy bandwidth | 168 GB/s (buffers ≥ 64 MiB) | was 1.5–9 GB/s |
| B1 target qubit (n = 26) | 6.39–6.71 ms for every target, no low-target penalty | **OQ10 resolved: no stride effect** (the earlier 5× was an artefact of the faulty clocks) |
| B2 scaling | flat ~0.012 ms up to 17 qubits; ×1.83 at 19, ×2.00 per qubit from 25 | same shape |
| B3 efficiency | general 1-qubit kernels 99.3–99.8% of copy | same |
| B4 specialised kernels | Z/T/CZ/CP 1.99×, SWAP 2.01×, X/CX/CCX/Rz 1.00× | SWAP was 3.57×: the general 4×4 path is no longer penalised, so SWAP's gain is just its halved bytes |
| B5 fusion (n = 22) | best width 2: QFT 1.84×, random 1.57× vs the general baseline; width 3 slower for QFT (0.88×); widths 4–5 much slower | was "m = 3, 2.08×" |
| B6 precision | complex128/complex64 time = 2.49–2.84 | **was 2.00–2.01.** With real bandwidth, consumer FP64 throughput shows, so complex128 isn't purely memory-bound |
| B7 comparison | see below | method changed |
| B8 capacity | 28 qubits complex64, 27 complex128 | same |
| VQE on GPU | max error 3.3e-5 Ha | was 6.3e-5 Ha (different run; both well inside 1.6e-3) |

**OQ9 revised: default fusion width = 2** (`fuse()` default changed from 3 to 2).

**B7 method fixed.** An audit found two biases in the first comparison, both favouring qsim:
1. Aer was asked for the full statevector, which it copies to Python (about 16% of Aer GPU time at
   24 qubits), while qsim returned nothing.
2. Aer's qubit truncation simulated only 21 of the 24 qubits of the random circuit.

Now every engine starts from |0…0⟩ and returns the same ⟨Z⟩ on qubit 0 (all engines agree within
2.2e-7), truncation is off, Aer runs with and without fusion, qsim with and without fusion (the
fusion pass inside the timed region), and every point is the median of 7 runs. Result at 24 qubits:
QFT 187 ms (qsim) vs 319 (cuStateVec, general per-gate API) vs 641 (best Aer GPU) vs 2307 (best Aer CPU).
Random circuit: 92 vs 125 vs 484 vs 913. From 18 qubits up, qsim was 1.2–1.7× faster than
cuStateVec used this way. Below 14 qubits cuStateVec was 1.7–2× faster. Aer CPU timings were noisy
on this laptop (one point's slowest run took 30× its fastest), so they're rough.

## Round 2: change requests W1–W8 (approved 2026-11-27)

Goal of the round: turn the validated simulator into an engine that a website, outside users and
other developers can build on. All eight CRs were approved together. Order of work: W5 rename,
W1 preparation, W2, W3, W4, rest of W5, W7, W6, W8, then W1 reporting once T4 results exist.

| CR | What | Changes earlier decisions | Effort |
|---|---|---|---|
| W1 | Re-run all benchmarks on a 16 GB T4. Every result gets a quick copy-bandwidth probe and is marked `"suspect"` if it is below 50% of the GPU's published bandwidth (T4: 320 GB/s, NVIDIA product brief, 2020). Results move to `bench/results/<gpu>/`. `bench/report.py` writes the README tables. Notebook fixed for Colab's CUDA 13 runtime. | OQ1: headline numbers come from the T4 | 6–8 h |
| W2 | `run(circuit, on_step=None)` hook; `views.py` with marginal probabilities, Bloch vectors, guarded snapshots and an entanglement hint (valid only for pure states); a versioned step-trace JSON. Marginals on the GPU use a reduction kernel. | changes the `Backend.run` signature | 8–12 h |
| W3 | Classical register, `measure`, `reset`, and gate conditions. Measurement reuses the expectation kernel for P(1) and `apply_diag` for the collapse. One host random generator feeds both backends so they give identical bits (except within ~1e-7 of a threshold). `run` returns the classical bits. Fusion and `inverse()` treat measurements as barriers. | extends the locked `Gate`/`Circuit` design | 16–24 h |
| W4 | Hand-written OpenQASM 2 import/export (including `gate` definitions and parameter expressions); new named gates `u`, `sx`, `sxdg`; optional OpenQASM 3 via Qiskit; Qiskit interop moves into the package. Export raises a clear error for what QASM 2 cannot express. Test circuits from MQT Bench (MIT). | reverses "no OpenQASM import" | 14–20 h |
| W5 | Rename the import package to avoid the PyPI `qsim` clash; numpy-only core install with `vqe` and `gpu-cuda12`/`gpu-cuda13` extras; public API and `py.typed`; changelog; trusted publishing (TestPyPI first, owner does the first real publish); wheel install test in CI. | removes matplotlib (OQ15) | 6–10 h |
| W6 | Run the NumPy backend in the browser through Pyodide (target 314.0.8, Python 3.14.2). Correctness check in CI with Node.js + Pyodide; browser timings measured by hand. | — | 8–12 h |
| W7 | Golden files: seeded GPU outputs plus SHA-256 of every kernel file, checked against the NumPy oracle on CPU CI. Optional GPU workflow on Kaggle through its API; the owner stores the token as a GitHub secret. | OQ17: CI still has no GPU, but kernel changes can no longer go unchecked | 6–10 h |
| W8 | General UCCSD VQE for n spin orbitals / n electrons, a general Hamiltonian file format and generator, and LiH with a small active space. VQE is judged against the exact energy of the same active-space Hamiltonian (CASCI); full FCI is reported alongside. | — | 12–20 h |

**W5 name decision (owner, 2026-11-27): change the name.** The import package becomes `qsimgpu`,
matching the PyPI distribution name `qsim-gpu` (free on PyPI when checked on 2026-11-27, while
`qsim` v0.22.0 and `qsimcirq` v0.22.1 are taken).

### W1 implementation notes (2026-11-27)
- Result files now live in one folder per GPU: `bench/results/<gpu-slug>/`. The laptop results
  moved to `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/`. File names in the sections above
  refer to files now in that folder.
- Every result records `bandwidth_check` (a 64 MiB copy). Published figures on file: T4, 320 GB/s
  (NVIDIA product brief, 2020). Below 50% of that, the file is marked `"suspect": true` and a
  warning is printed. GPUs without a published figure on file (such as the laptop's) get no gate.
  Older files whose driver reported a faulty clock state also count as suspect.
- README result tables are written by `python -m bench.report`, which refuses suspect files.
- T4 runs go through Kaggle (`scripts/kaggle_run.py`) using the owner's own Kaggle CLI login; the
  key is never read or stored by the project.

## W1 result: Tesla T4 run on Kaggle (2026-11-27)

One private Kaggle run (T4, driver 580.178.04, CUDA runtime 13.0, Python 3.13.16, commit `85221cd`)
ran the full suite: **343 passed, 1 skipped on the T4** (`bench/results/tesla-t4/test_log.txt`).
Every result passed the bandwidth check (64 MiB copy at 195–236 GB/s, 61–74% of the published
320 GB/s). Headline figures now come from `bench/results/tesla-t4/`; the README tables are generated
from it.

- **Capacity:** 30 qubits complex64, 29 complex128 (31 / 30 ran out of memory). This confirms the arithmetic.
- **Efficiency:** H gate on 26 qubits 234 GB/s vs a 230 GB/s copy in the same run, so the copy kernel is a
  yardstick rather than a hard ceiling (an in-place gate can be marginally faster than a two-buffer copy).
- **Specialised kernels** about 2× where bytes halve; **fusion** best at width 2 (QFT 1.92×) or 3
  (random 1.75×, 1.71× at width 2: within noise). Default stays 2.
- **Comparison:** qsim faster than cuStateVec's general per-gate API from 20 qubits (1.3–2.2×), slower at 18;
  faster than Aer GPU from 20 qubits (2.2–7.0×). Engines agree on ⟨Z0⟩ within 2.2e-7.
- **Open anomaly:** complex128/complex64 time ratio is 3.6–3.9 at 20–24 qubits but 2.0 at 26. The power-cap
  flag (0x4) was set during this benchmark. Not explained yet; to re-run in isolation.
- **Tooling fix:** the Aer-GPU environment has no CuPy, so its results were filed under `cpu-only`. The folder
  name now falls back to the `nvidia-smi` GPU name, and the two files were moved to `tesla-t4/`
  (their recorded environment shows Tesla T4).
- The Kaggle image prints a harmless `sitecustomize` warning (`wrapt` missing) inside the uv environments.

## W2 result: step-by-step execution and state views (2026-11-27)

- `Backend.run(circuit, on_step=None)` on both backends; `qsimgpu/views.py` with
  `marginal_probabilities`, `bloch_vector`, `entanglement_hint`, `snapshot`, `step_trace`
  (`trace_version: 1`) and `trace_max_difference`. New CUDA kernel `marginal_probs`.
- Measured on the RTX 3050 Laptop (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/w2_steps.json`):
  the `on_step` hook with an empty callback costs nothing measurable (±0.5 µs per gate on the GPU,
  within run-to-run noise). A full trace of a 20-gate, 10-qubit circuit takes 30 ms on NumPy and
  65 ms on the GPU: at 10 qubits each of the 30 Bloch-vector passes per step is dominated by launch
  overhead, so the GPU only pays off for larger states.
- NumPy vs CUDA: traces differ by at most 2.4e-7 (20 random circuits), and marginals of 20-qubit
  states by 8.2e-11. New tolerance `VIEW_ATOL_COMPLEX64 = 1e-5`.
- Later list: a single kernel for all three Bloch components of all qubits (one pass instead of 3n)
  would make GPU traces faster.

## W3 result: mid-circuit measurement, reset and classical conditions (2026-11-27)

- IR: `Circuit(n, n_clbits=...)`, `measure(q, c)`, `reset(q)`, `c_if(clbits, value)` (a condition on
  the most recent gate; bit j of the value = clbits[j]). `Gate` gained `clbits` and `condition`
  fields; `inverse()` and fusion refuse to cross them.
- `qsimgpu/execution.py`: the shared run loop, `run_shots`, `has_mid_circuit_measurement`.
  `Backend.run` now returns the classical bits and takes `rng`; backends gained `measure_qubit`,
  `reset_qubit` and a `classical_bits` attribute (recorded per step in traces).
- GPU measurement reuses existing kernels: P(1) = (1 − ⟨Z⟩)/2 from the expectation kernel, and the
  collapse is a single `apply_diag` pass.
- Same bits on both backends: checked over 25 seeds on random circuits with 5 mid-circuit
  measurements (`test_numpy_and_cuda_give_identical_bits_for_the_same_seed`).
- Statistics: measuring |+⟩ 2000 times passes a two-sided exact binomial test at p > 1e-3.
- New demos `teleport.py` and `bitflip.py`, both tested on both backends.
- Cost (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/w3_shots.json`, 12 qubits, 60 gates,
  1000 shots): measurement only at the end takes 1.6 ms (NumPy) / 1.7 ms (GPU). One mid-circuit
  measurement forces a re-run per shot: 1687 ms / 1559 ms.

## W4 result: OpenQASM 2 import and export (2026-11-27)

- `qsimgpu/qasm.py`: hand-written tokenizer and recursive-descent parser, `from_qasm2` / `to_qasm2`.
  No new dependency.
- New named gates `u` (Qiskit U = OpenQASM u3), `sx`, `sxdg`, tested against Qiskit's matrices and
  on both backends' kernels. Mapping: u1 → p, u2 → u(π/2, φ, λ), cu1 → cp, controlled forms → base
  gate plus a control, rzz/rxx → exact decompositions. `id`, `u0` and `barrier` are dropped on
  purpose, since they don't change the state.
- Qiskit writes OpenQASM 2 against an extended qelib1.inc that declares, without defining, u0, cu,
  c3x, c3sqrtx, c4x, rccx and rc3x. These are built in (rccx/rc3x by their standard gate sequences),
  and each is checked against Qiskit's own matrix.
- Expression precedence follows ordinary maths (`-pi^2 = -(pi^2)`, `2^-1 = 0.5`). An earlier draft
  had unary minus binding tighter than `^`; a test caught it.
- Export uses qelib1 names and raises `QasmExportError` for `unitary` gates, more than one control
  (except ccx and cswap), and conditions that aren't a whole classical register.
- Tests: exact round trips (`to_qasm2` → `from_qasm2` gives identical gates), 22 OpenQASM files
  generated with MQT Bench 2.4.0 (MIT) in `tests/data/qasm/`, and 15 random Qiskit circuits, all
  matching Qiskit's statevector. The comparison allows a global phase, because Qiskit's own
  custom-gate definitions can drop one when written as OpenQASM 2. Malformed inputs produce errors
  with the right line.
- `qsimgpu/interop/qiskit.py` (optional): `to_qiskit` (conditions become `if_test` blocks),
  `from_qiskit` (through OpenQASM 2) and `from_qasm3` (through `qiskit.qasm3`, which needs
  `qiskit-qasm3-import`). `tests/qiskit_bridge.py` now only keeps the Aer helper.

## W5 result: packaging (2026-11-27)

- Version 0.2.0 with `CHANGELOG.md` (semantic versioning). Core install = NumPy only; `import qsimgpu`
  loads neither CuPy nor scipy (tested in a subprocess). Extras: `vqe`, `gpu-cuda12`, `gpu-cuda13`
  (`gpu` kept as CUDA 12), `compare`, `chem`, `dev`. matplotlib dropped (OQ15). VQE raises a clear
  error naming the `vqe` extra if scipy is missing.
- The CuPy wheels: `cupy-cuda12x` stays right for CUDA 12 drivers. Colab's current runtime ships
  CUDA 13 and `cupy-cuda13x` (checked 2026-11-27), hence the separate `gpu-cuda13` extra. Both
  include the `[ctk]` headers that kernel compilation needs.
- Public API in `qsimgpu/__init__.py` (`__all__`); `CudaBackend` is loaded on first use. `py.typed` is shipped.
- CI builds the wheel, checks it contains the kernels and `py.typed`, installs it into a clean
  environment and runs a Bell circuit (`scripts/check_wheel.py`). Checked locally too: the wheel has
  32 files, and neither it nor the sdist contains any untracked local files.
- `release.yml`: a `v*` tag publishes to TestPyPI; real PyPI only when started by hand. Both use
  trusted publishing, which the owner must enable once on test.pypi.org and pypi.org (publisher =
  this repository + `release.yml`, environments `testpypi` and `pypi`). No tokens in the repository.

## W7 result: golden files, so CPU-only CI checks GPU results (2026-11-27)

- `scripts/make_goldens.py` runs 26 fixed circuits on the GPU:
  - random circuits at 2–12 qubits;
  - QFT at 4/8/12;
  - Grover;
  - GHZ;
  - one circuit per gate family (diagonal, permutation, general, including multi-controlled
    gates, controlled SWAP and a `unitary`);
  - fused versions at widths 2–5, which exercise `apply_k2`…`apply_k5`.

  Each runs in complex64 and complex128 (the family cases and one random case also through the
  general kernels), giving 52 stored states, 1.5 MB in `tests/data/goldens/` (generated on the
  RTX 3050 Laptop).
- `metadata.json` holds each circuit as JSON (`views.circuit_to_dict`), the environment, and the
  SHA-256 of `qsimgpu/kernels/*.cu`, `*.cuh` and `qsimgpu/backends/cuda_backend.py`.
- `tests/test_goldens.py` (CPU) recomputes every circuit with the NumPy oracle and compares, within
  the measured tolerances. If any watched file's hash differs, it fails with a message to regenerate
  the goldens on a GPU. Checked: appending a comment to `gates.cu` makes it fail, and restoring the
  file makes it pass.
- Honest limit: CI still doesn't execute the kernels. It proves that the stored outputs of the
  current kernel code are correct, and that the code hasn't changed since.
- Optional `gpu-kaggle.yml` (manual only): pushes a private Kaggle script in tests-only mode, waits,
  downloads the log and fails if any test failed. It needs the owner to add a secret
  `KAGGLE_API_TOKEN` and a variable `KAGGLE_USERNAME` in the GitHub settings (the Kaggle CLI reads
  the token from that environment variable). The workflow hasn't been run yet.
