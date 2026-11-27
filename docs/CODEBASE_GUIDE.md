# Codebase guide

A walk through every folder and file: what it does, why it's built that way, and the
problems that came up while building it. Read it top to bottom once, then use it as a
reference before an interview.

---

## 1. The big picture

```
 algorithms/   grover.py, qft.py, shor15.py, vqe.py      "the programs we run"
      │ uses only ↓
 Backend       reset / apply / run / sample / expectation  "one interface, two engines"
      │                         │
 numpy_backend.py          cuda_backend.py ── kernels/*.cu (CUDA C, compiled at run time)
 (reference, slow)         (GPU, fast)
      │ both consume ↓
 circuit.py    Gate records + Circuit builder               "what to compute, as plain data"
 gates.py      exact matrices + gate families
 passes/       fusion.py: rewrites a Circuit before a backend runs it
```

Each layer only depends on the layers below it. An algorithm never imports CuPy: it builds a
`Circuit` and hands it to whatever `Backend` it was given. That's why every algorithm test runs
twice, once on NumPy and once on the GPU, with the same code.

If you know web backends, a rough mapping:

| Here | Web backend equivalent |
|---|---|
| `circuit.py` | request schema / DTO: plain data, validated on creation |
| `Backend` protocol | repository interface |
| `numpy_backend.py` | an in-memory fake you trust completely |
| `cuda_backend.py` | the real database driver |
| `kernels/*.cu` | stored procedures running inside the database |
| `algorithms/` | business logic / route handlers |
| `tests/` | unit + integration tests |
| `bench/` | load tests, with every result saved |

---

## 2. File by file

### Root

| File | What and why |
|---|---|
| `pyproject.toml` | Package definition (like `package.json`). Optional extras: `gpu` (CuPy), `dev` (pytest, hypothesis, ruff, mypy), `compare` (Qiskit, Aer), `chem` (PySCF, OpenFermion, only for the offline data script). Strict mypy and ruff settings live here too. |
| `README.md` | What the project is, results with their caveats, how to run it, limitations. |
| `LICENSE` | MIT. |
| `.github/workflows/ci.yml` | GitHub Actions runs lint, type check and the CPU tests on Python 3.12 and 3.14. Free runners have no GPU, so GPU tests are skipped there (they skip themselves). |
| `.gitignore` | Virtual environments and tool caches. |

### `qsim/gates.py`, the gate catalogue
- `SPECS`: for every gate name, how many targets, controls and parameters it takes, which base gate
  it applies (`cx` applies `x`), and its **family**: `diagonal` (only multiplies amplitudes by
  phases), `permutation` (only moves amplitudes) or `general` (mixes them). The family decides which
  GPU kernel can run it.
- `base_matrix(name, params)`: the exact 2×2 (or 4×4 for SWAP) matrix, Qiskit conventions.
  Rz(θ) = diag(e^{−iθ/2}, e^{iθ/2}), which differs from P(θ) only by a global phase.
- `inverse_name_params`: s ↔ sdg, t ↔ tdg, rotations negate their angle, H/X/Y/Z/CX/SWAP undo
  themselves.
- Multi-target matrices are little-endian over the target list: bit i of the row index belongs
  to `targets[i]`.

### `qsim/circuit.py`, circuits as plain data
- `Gate` is a frozen dataclass: name, targets, controls, params, and a matrix only for `"unitary"`.
  It validates itself on creation (right number of targets/params, no qubit used twice, unitary
  matrices really unitary).
- **Canonical encoding:** targets and controls are separate fields. CNOT with control 0 and target
  1 is `Gate("cx", targets=(1,), controls=(0,))`. Any single-qubit gate may carry extra controls,
  which is how Grover's many-controlled Z is written: `Gate("z", (n-1,), controls=(0..n-2))`.
- **A subtle Python issue:** a frozen dataclass normally gets an automatic `__eq__` and `__hash__`.
  With a NumPy array inside, `==` returns an array instead of True/False and hashing fails. So
  `Gate` defines both by hand: compare the matrix with `np.array_equal`, and hash only the
  hashable fields.
- `Circuit` holds the gate list and has builder methods (`h`, `cx`, `cp`, `mcz`, `unitary`, …)
  that return `self`, so calls chain: `Circuit(2).h(0).cx(0, 1)`. Also `inverse()` (reverse order,
  invert each gate), `depth()` (layers if every gate starts as early as possible) and `count_ops()`.

### `qsim/backends/base.py`, the shared interface
- `Backend` is a `typing.Protocol`: the methods every simulator provides (`reset`, `apply`,
  `run`, `statevector`, `probabilities`, `sample`, `expectation_pauli`,
  `expectation_hamiltonian`, `amplitude_norm`).
- `pauli_masks("XIZY", n)` turns a Pauli string into two bit masks plus a count. That turns
  "apply a Pauli string" into bit operations (section 3.4).
- **Pauli order:** the leftmost character is the highest qubit, the same way bit patterns are
  written. A test checks `"IZ"` measures qubit 0 and `"ZI"` measures qubit 1.

### `qsim/backends/numpy_backend.py`, the reference ("oracle")
- **Goal: obviously correct, not fast.** Double precision (complex128), so it's more precise
  than the GPU it judges.
- The state is stored with shape `(2, 2, …, 2)`, one axis per qubit, with qubit k on axis n−1−k.
  Then the flattened index is exactly the little-endian integer.
- **One routine for every gate** (`apply_matrix`): take the slice where all control axes equal 1,
  contract the target axes with the gate matrix (`np.tensordot`), and put the result back.
  The tricky detail: reshaping a 2^k × 2^k matrix to `(2,)*2k` orders axes most-significant
  first, so the target axes have to be passed in reverse.
- Pauli expectations are computed the slow, literal way: apply the Pauli matrices to a copy and
  take an inner product. That's deliberately different from the GPU's bit-mask formula, so the two
  can catch each other's bugs.

### `qsim/backends/cuda_backend.py`, the GPU engine
- The state is one CuPy array of 2ⁿ amplitudes in GPU memory.
- Kernels are compiled at run time with NVRTC through `cupy.RawModule`, one compile per file and
  precision, cached with `functools.cache`. No nvcc and no build step.
- `apply(gate)` picks a kernel. With `specialized=True` (default): phase gates → `apply_phase`,
  Rz → `apply_diag`, X/CX/CCX → `apply_x`, SWAP → `apply_swap`. Everything else (H, Y, Rx, Ry, fused
  unitaries) → the general kernels. With `specialized=False`, everything takes the general path.
  This switch is how the before/after measurement was done.
- `sample()` uses two passes over 1024-amplitude chunks (section 3.6). `sample_cumsum()` is the
  simple version, kept for comparison.
- `pauli_value()` / `expectation_pauli()` use one reduction kernel, summed in double precision.
- `amplitude_norm()` reuses the same kernel with empty masks: ⟨ψ|ψ⟩ without a temporary array.
- `statevector()` and `probabilities()` refuse above 26 qubits, because copying 2ⁿ values to the
  host would be huge.

### `qsim/kernels/`, the CUDA C code
| File | Kernels |
|---|---|
| `common.cuh` | `cplx` (complex<float> or complex<double>), 64-bit `idx_t`, and `insert_zero_bits` |
| `gates.cu` | `copy_state` (bandwidth yardstick), `apply_1q` (general 2×2), `apply_k1`…`apply_k5` (general 2^k×2^k, one compiled version per k), `apply_phase`, `apply_diag`, `apply_x`, `apply_swap` |
| `measure.cu` | `chunk_prob_sums`, `locate_in_chunk` (two-pass sampling) |
| `expect.cu` | `pauli_partial` (Pauli expectation partial sums per block) |
| `__init__.py` | `build_source(file, dtype)`: prepends `typedef float real_t;` or `typedef double real_t;` plus the common header. The same source builds both precisions, so the precision study compares identical code. |

All kernels use **grid-stride loops** (each thread handles items t, t + stride, …), so one
launch shape works for any problem size.

### `qsim/passes/fusion.py`
`fuse(circuit, max_qubits=m)` walks the gates in order and grows a block while the union of
their qubits stays within m. When the next gate would exceed m, the block becomes one `unitary`
gate whose matrix is computed by running the block on a tiny NumPy simulator, one column per
basis state. A one-gate block is left alone so it keeps its specialised kernel.

### `qsim/algorithms/`
| File | What it builds | Checked by |
|---|---|---|
| `qft.py` | H + controlled-phase ladder + swaps; `qft_gates()` can target any qubit list (Shor uses it on 8 of 12 qubits) | QFT of each basis state equals e^{2πixk/N}/√N; a comb with spacing r peaks at multiples of N/r |
| `grover.py` | oracle (X-wrapped multi-controlled Z) + diffusion; `optimal_iterations()` compares floor vs round of π/4·√N | n=2 certain after 1 round; n=3 gives 0.945 after 2; matches sin²((2k+1)θ) |
| `shor15.py` | 8 input + 4 output qubits, controlled multiplication by a^(2^j) mod 15 as swap/X networks, inverse QFT, continued fractions + gcd | every multiplier circuit checked against a^p·y mod 15 for all y; a=7 peaks at 0/64/128/192; recovers r and 3 × 5 |
| `vqe.py` | H₂ energy scan: Hamiltonian loading, exact diagonalisation, Pauli-rotation circuits, UCCSD generators, SciPy optimiser | each building block vs `scipy.linalg.expm`; whole scan within chemical accuracy |

### `scripts/gen_h2_hamiltonians.py` and `data/h2_hamiltonians.json`
Run once, offline, in a separate environment with PySCF and OpenFermion. For each of 45 bond
lengths: Hartree–Fock and full CI in PySCF, molecular integrals → fermionic Hamiltonian →
Jordan–Wigner (4 qubits) and symmetry-reduced Bravyi–Kitaev (2 qubits). Before writing, it
checks that the reduced Hamiltonian's lowest eigenvalue equals the FCI energy. No coefficient is
typed by hand.

### `tests/`
| File | Covers |
|---|---|
| `conftest.py` | skips `@pytest.mark.gpu` tests automatically when there's no usable GPU |
| `tolerances.py` | every tolerance in one place, with where its value came from |
| `circuits.py` | seeded random circuits (whole gate set, sometimes extra controls), random states, random Pauli strings |
| `qiskit_bridge.py` | converts a qsim circuit to Qiskit with **no** reordering or phase fixes, so any disagreement is a real bug |
| `test_gates.py`, `test_circuit.py` | matrices, identities, validation, inverse, equality/hash |
| `test_numpy_backend.py` | bit order, Bell/GHZ, H-H with and without a measurement, invariants, sampling, Pauli order |
| `test_aer_crosscheck.py` | oracle vs Aer on random circuits up to 12 qubits |
| `test_cuda_backend.py` | GPU vs explicit matrices and vs oracle, both kernel paths, sampling chi-square, expectations |
| `test_algorithms.py` | GHZ, H-H, Grover, QFT, Shor on both backends |
| `test_fusion.py` | fused circuit = unfused circuit, m = 1..5, CPU and GPU |
| `test_vqe.py` | Pauli rotations, UCCSD generators, data cross-checks, chemical accuracy over the grid |

### `bench/`
| File | Question |
|---|---|
| `run_env.py` | records GPU, driver, CUDA, NVRTC, library versions, git commit and live clock state; `time_gpu()` (CUDA events, warm-up, median of 7); `save_result()` |
| `run_tolerance.py` | how far does complex64 drift from the oracle? (sets the test tolerance) |
| `run_b0_bandwidth.py` | B0: copy bandwidth, the yardstick |
| `run_b1_target.py` | B1: does the target qubit change gate time? |
| `run_b2_scaling.py` | B2: time vs number of qubits |
| `run_b3_efficiency.py` | B3: gate speed as a fraction of copy speed, measured back to back |
| `run_b4_specialised.py` | B4: specialised vs general kernels |
| `run_b5_fusion.py` | B5: fusion width m = 1..5 |
| `run_b6_precision.py` | B6: complex64 vs complex128 |
| `run_b7_compare.py` | B7: same circuits on qsim, Aer CPU/GPU, cuStateVec |
| `run_b8_scale.py` | B8: grow n until out of memory |
| `run_vqe_h2.py` | the H₂ energy curve |
| `run_all.py` | everything above in order (`--big` for a 16 GB GPU) |
| `results/*.json` | every measured number, each with its environment record |

### `docs/`
`DECISIONS.md` (every choice, with the measurement or reason behind it), `ENVIRONMENT.md` (exact
versions and how to recreate them), `WRITEUP.md` (one-page summary), and this guide.

### `notebooks/colab_run.ipynb`
Clones the repo on a Colab GPU, installs it, runs all tests and `bench.run_all --big`, and
downloads the results.

### `lessons/`
`lesson0.py` builds a 1- and 2-qubit simulator in plain Python. `lesson1_cuda.py` shows the
GPU side: CuPy arrays, a raw CUDA kernel, and the pair-index trick on the GPU. `teaching.md`
explains the concepts and every gate.

---

## 3. The ideas worth explaining in an interview

### 3.1 Why 2ⁿ, and why it's memory-bound
n qubits → 2ⁿ amplitudes, because entangled qubits can't be described one at a time. A gate
touches every amplitude once with a handful of multiply-adds: about 1 flop per byte moved.
GPUs can do far more arithmetic per byte than that, so memory bandwidth sets the speed.
**Roofline lower bound:** one pass moves 2 × 8 × 2ⁿ bytes in complex64, so the minimum time is
that divided by the achievable bandwidth (measured by B0, not taken from a spec sheet).

### 3.2 The pair-index trick
A gate on qubit k mixes amplitude pairs whose indices differ only in bit k. Thread t computes
its pair directly:
```
low  = t & ((1 << k) - 1)
i0   = ((t >> k) << (k + 1)) | low     # insert a 0 at bit k
i1   = i0 | (1 << k)
```
No two threads share an amplitude, so there are no races, no locks and no synchronisation.
`insert_zero_bits` generalises this to several fixed bits (target + controls). That's how a
controlled gate launches threads **only** for pairs whose controls are 1, instead of launching
all of them and letting half return early.

### 3.3 64-bit indices
At 30 qubits the byte offset passes 2³²: an `int` index silently wraps around. Every kernel
uses `long long` for indices and masks.

### 3.4 Pauli expectation in one pass
For a Pauli string P, X and Y flip bits and Z and Y add signs, so
P|i⟩ = i^{n_Y} · (−1)^{popcount(i & z_mask)} · |i ⊕ x_mask⟩. That turns ⟨ψ|P|ψ⟩ into one sum over i
of conj(ψ[i ⊕ x_mask]) · sign · ψ[i]: one read pass, no copy of the state. The result must be real.
The tests check the imaginary part is ~0, which catches sign mistakes.

### 3.5 Summing in double
Each block of threads writes one partial sum in double precision, even for a complex64 state.
A float32 running total over 2²⁸ terms would lose digits.

### 3.6 Sampling without a second copy of the state
The obvious approach (cumulative sum of all 2ⁿ probabilities) needs 8 bytes per amplitude in
float64, more than the complex64 state itself. Instead: pass 1 sums each 1024-amplitude chunk,
the host picks a chunk per shot from those 2ⁿ/1024 totals, and pass 2 walks inside the chosen chunk.
Both methods pass the same chi-square test.

### 3.7 Bit order and global phase
Qubit 0 is the least significant bit, the same as Qiskit, so results compare with Aer **without
any reordering**. The test bridge also applies no global-phase fix, since the gate definitions match
exactly. One trap: Rz and P differ only by a global phase *uncontrolled*, but once controlled the
difference becomes a relative phase that changes results. There's a test for exactly that.

### 3.8 The optimisation story (one measured step at a time)
1. **Baseline:** general kernels; the H gate already ran at 99.5–99.8% of copy speed
   (168 GB/s copy on the RTX 3050 Laptop).
2. **Specialised kernels:** removing multiplications (X, CX, Rz) gained nothing. Reading fewer
   bytes (phase gates touch half the state, CZ/CP a quarter, SWAP half) gained 1.99–2.01×.
   That's the clearest proof the kernels are memory-bound.
3. **Fusion, first try:** fewer passes but slower. The kernel's per-thread arrays were sized for the
   biggest case and spilled out of registers into slow local memory, and fused blocks lose the
   savings of specialised kernels.
4. **Fusion, fixed:** one compiled kernel per matrix size (C++ template, unrolled, matrix in shared
   memory). Width 2 then ran the QFT 1.84× and a random circuit 1.57× faster than the
   general-kernel baseline. Widths 4–5 are still slower: too much arithmetic per byte and too
   many registers.
5. **Precision:** complex128 costs 2.5–2.8× per gate, more than the 2× byte ratio, because this
   consumer GPU's double-precision units are slow. And one qubit less fits.
6. **Fair comparison:** with every engine returning the same number and simulating the full
   state, qsim ran 1.2–1.7× faster than cuStateVec's general per-gate API from 18 qubits up,
   and 3.4–5.3× faster than Aer GPU at 24 qubits. Below 14 qubits cuStateVec was faster
   (qsim pays more Python overhead per gate).

### 3.9 VQE in one paragraph
A circuit with adjustable angles prepares a trial state, the simulator returns its exact energy,
and a classical optimiser (L-BFGS-B) adjusts the angles. For H₂: start from Hartree–Fock |0011⟩
and apply three excitations (two singles, one double). Each is exp(θ(T − T†)), built from
Jordan–Wigner fermion matrices and split into Pauli strings numerically. The strings inside one
excitation commute, so one Trotter step is exact. Each exp(iφP) becomes a basis change, a CX ladder,
one Rz and the undo. The singles stay ~0 by symmetry, which was measured rather than assumed.

---

## 4. Problems hit while building it, and what fixed them

| Problem | Cause | Fix |
|---|---|---|
| Kernel compile failed: "Failed to find CUDA headers" | plain `cupy-cuda12x` doesn't ship the CUDA headers NVRTC needs | install `cupy-cuda12x[ctk]` |
| Would CUDA 12.9 libraries work with a driver that only supports 12.2? | NVRTC compiles straight to machine code for the GPU (sm_86), so the driver never has to JIT newer PTX | verified by the smoke test; recorded in `docs/ENVIRONMENT.md` |
| Compile error: "Thrust requires at least C++17" | CuPy's complex header pulls in Thrust | compile kernels with `-std=c++17` |
| Aer-GPU and cuQuantum wouldn't install together | conflicting dependencies; Aer-GPU also replaces `qiskit-aer` | a separate environment for each comparison engine |
| `Gate == Gate` crashed when gates held a matrix | dataclass-generated `__eq__` compared NumPy arrays | hand-written `__eq__` / `__hash__` |
| Comb-QFT test failed for r = 2 | test bug: a comb starting at 3 with spacing 2 isn't periodic from index 1 | start the comb at `3 % r` |
| Continued fractions gave r = 2 for a = 7 | measured 128/256 = 1/2 shares a factor with the true period | if a^r ≢ 1, try 2r and 4r |
| The 4-qubit Hamiltonian's lowest eigenvalue isn't the H₂ answer in general | Jordan–Wigner covers every electron count | diagonalise only the 2-electron sector; test it equals FCI |
| Benchmarks showed ~8 GB/s on a GPU rated much higher | the laptop driver reported "System is not in ready state" (faulty clock state); a reboot fixed it | record live clock state in every JSON, flag such runs, compare only within one run, re-run later |
| The first comparison flattered qsim | Aer returned the full statevector while qsim returned nothing, and Aer's qubit truncation simulated only 21 of 24 qubits | every engine returns the same ⟨Z0⟩ value; truncation off; 7 repeats; engines checked to agree within 2.2e-7 |
| Small buffers showed 100+ GB/s "bandwidth" | they fit in the 2 MB L2 cache | define achievable bandwidth on buffers ≥ 64 MiB only |
| Fusion made circuits slower | register spills in a one-size-fits-all kernel | one templated kernel per matrix size |
| The GPU ran 4-qubit VQE slower than NumPy | each kernel launch costs more than updating 16 amplitudes | expected: a GPU only pays off for large states |

---

## 5. Likely interview questions

**Why not just use Qiskit?** The point was to understand where the time goes. Building it
showed it's memory-bound and which optimisations help and why. Qiskit Aer is used as an
independent correctness check and as a performance comparison.

**How do you know it's correct?** Two independent references (own NumPy oracle, Qiskit Aer),
explicit full-matrix checks for small n, invariants (norm, circuit then inverse), known algorithm
answers, and a tolerance measured rather than guessed.

**Why is the GPU not 100× faster than the CPU?** Both are limited by memory bandwidth here, not
arithmetic, so the speed-up is roughly the ratio of their memory bandwidths. Also, small circuits
are dominated by kernel-launch overhead.

**What was the hardest bug?** Fusion that made things slower despite doing fewer passes. The
pass count went down, but the kernel's local arrays spilled out of registers, adding memory traffic
to a memory-bound kernel. Compiling one kernel per matrix size fixed it.

**Does this show quantum advantage?** No. It's a classical simulation, and H₂ in a minimal
basis is classically solvable, which is why it's the validation case.

**Why complex64?** Its error (~1e-7) is far below what the algorithms need, and it halves memory
traffic and adds one qubit of capacity. complex128 is available and measured at 2.5–2.8× the cost.

**What would you do next?** Run the benchmarks on a 16 GB T4 (29–30 qubits). Make fusion
smarter (keep diagonal runs diagonal, allow reordering gates on disjoint qubits). Add shared-memory
tiling for low qubits. Implement general modular exponentiation for Shor.
