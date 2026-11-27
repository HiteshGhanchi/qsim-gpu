# qsim-gpu

A quantum circuit simulator written from scratch: Python on top, hand-written CUDA kernels
underneath. It is checked against a NumPy reference and against Qiskit Aer, and used to run
Grover search, the quantum Fourier transform, Shor's algorithm for N = 15, and a VQE
calculation of the H₂ molecule.

The question behind the project: **where does a GPU statevector simulator spend its time,
and how close can it get to the limit set by memory bandwidth?**

## Why simulating qubits is hard

n qubits are described by 2ⁿ complex numbers (amplitudes), one for every bit pattern. Every
extra qubit doubles the memory and the work: 20 qubits need 8 MB in single precision,
30 qubits need 8 GB. A gate on one qubit only mixes pairs of amplitudes whose indices differ in that qubit's bit,
so it is one streaming pass over the whole array with a few multiplications per element.
That makes the simulator **memory-bound**: the speed of a gate is decided by how fast the
GPU can read and write its memory, not by arithmetic. Most of the engineering here follows
from that one fact.

This is a simulator, not a quantum computer. It doesn't show any quantum speedup. Its value
is in understanding why classical simulation is expensive, and in making it fast and correct.

## What is in it

| Part | What it does |
|---|---|
| `qsim/gates.py` | Exact gate matrices (Qiskit conventions) and which memory-access family each gate belongs to |
| `qsim/circuit.py` | A circuit as plain data: validated `Gate` records, builder methods, `inverse()`, `depth()` |
| `qsim/backends/numpy_backend.py` | The reference simulator: simple, double precision, used to judge everything else |
| `qsim/backends/cuda_backend.py` + `qsim/kernels/*.cu` | The GPU simulator: CUDA C kernels compiled at run time through CuPy |
| `qsim/passes/fusion.py` | Gate fusion: merge consecutive gates into one matrix, so the state is streamed fewer times |
| `qsim/algorithms/` | GHZ, Grover, QFT, Shor (N = 15), VQE for H₂ |
| `bench/` | Benchmarks; every result is stored as JSON with the GPU, driver, CUDA and library versions |
| `tests/` | 300+ tests: CPU tests run anywhere, GPU tests skip themselves without a GPU |
| `docs/` | Decisions with evidence, environment, interview write-up, a file-by-file guide |
| `lessons/` | Beginner notes on qubits, gates and entanglement, with a small plain-Python simulator |

## How correctness is established

- **Two independent references.** The NumPy backend applies every gate with one general
  tensor contraction. Qiskit Aer is a separate, widely used simulator. The NumPy backend
  matches Aer exactly (tolerance 1e-9, no reordering or phase fixes) on random circuits up to
  12 qubits, and the GPU backend matches NumPy on random circuits up to 20 qubits.
- **Explicit matrices.** For small n, every gate on the GPU is compared with its full 2ⁿ × 2ⁿ
  matrix, built bit by bit.
- **Invariants.** The norm stays 1 after every gate, and running a circuit followed by its
  inverse returns the starting state.
- **Known answers.** H-H gives back |0⟩. Grover's success probability follows sin²((2k+1)θ).
  The QFT of a basis state matches the Fourier formula. Shor recovers period 4 for a = 7 and the
  factors 3 × 5. VQE matches exact diagonalisation of the H₂ Hamiltonian at all 45 bond
  lengths.
- **Measured tolerance.** The single-precision tolerance (1e-5) came from a measurement. The
  worst observed error over random depth-100 circuits up to 20 qubits was 5.5e-7.
- **Engines agree.** In the simulator comparison, qsim, Qiskit Aer and cuStateVec return the
  same ⟨Z⟩ value within 2.2e-7 on every circuit.

## Results

Every number below comes from a JSON file in `bench/results/`, which also records the GPU,
driver, CUDA and library versions, the live GPU clocks and the git commit.

**Hardware:** NVIDIA RTX 3050 Laptop GPU (4 GB, Ampere), driver 535, CUDA runtime 12.9,
complex64 unless stated. It's a laptop part: during the runs the driver reported software power
and thermal capping (recorded in each JSON), so expect a few percent of variation between runs.

**Memory bandwidth is the limit** (B0–B3). A plain copy of the state reaches 168 GB/s. One H
gate on a 26-qubit state takes 6.4 ms, which is 99.5–99.8% of the copy speed for the general
single-qubit kernel, so there is essentially nothing left to gain on that gate. From 19 qubits
on, each extra qubit roughly doubles the time per gate (×1.83 at 19, ×2.00 from 25 on). Below 18 qubits the state fits in the 2 MB L2
cache and each gate costs a flat ~0.012 ms, which is mostly kernel-launch overhead. The target
qubit makes no systematic difference (6.39–6.71 ms across all 26 targets).

**Moving fewer bytes is what helps** (B4, 24 qubits, both paths timed in the same run):

| Gate | Speedup over the general kernel | Reason |
|---|---|---|
| Z, T | 1.99× | only amplitudes with the target bit set are read |
| CZ, CP | 1.99× | only the \|11⟩ quarter is touched |
| SWAP | 2.01× | only \|01⟩ and \|10⟩ move |
| X, CX, CCX, Rz | 1.00× | fewer multiplications, same bytes: no gain |

**Gate fusion** (B5, 22 qubits). Merging consecutive gates on up to 2 qubits gave the best
results: the QFT ran 1.84× faster than the general-kernel baseline (37 ms vs 69 ms) and a depth-100 random
circuit 1.57× faster (21 ms vs 32 ms). Wider fusion cuts more passes but is slower on this GPU:
at width 4–5 a 16×16 or 32×32 complex matrix per amplitude group is no longer memory-bound.
The first fusion kernel was slower than no fusion at all, because its per-thread arrays didn't
fit in registers. Compiling one kernel per matrix size fixed it.

**Precision** (B6). complex128 costs one qubit of capacity and takes 2.5–2.8× as long per gate
(not 2×). The extra time beyond the 2× byte ratio comes from this consumer GPU's slow
double-precision arithmetic. complex64's error against the double-precision reference stays
below 1.2e-7, so it's the default.

**Capacity** (B8). On the 4 GB GPU: 28 qubits in complex64 and 27 in complex128, each checked by
norm and by sampling a GHZ state. 29 qubits (a 4 GB state) runs out of memory.

**VQE for H₂.** Across 0.30–2.50 Å the VQE energy stays within 1.4e-10 Hartree of exact
diagonalisation on the NumPy backend, and within 3.3e-5 Hartree on the GPU in single precision.
Chemical accuracy is 1.6e-3. The minimum is at 0.75 Å, −1.137117 Hartree.

**Compared with other simulators** (B7, `bench/results/b7_compare_*.json`). The same circuits in single
precision on every engine. Each run starts from |0…0⟩ and returns the same single number (⟨Z⟩ on qubit 0),
which all engines agree on within 2.2e-7. Aer's qubit truncation is off, so every engine simulates
the full state. Median of 7 runs. Times in ms:

| Circuit | Qubits | Aer CPU (16 threads)¹ | Aer GPU² | cuStateVec³ | qsim | qsim + fusion |
|---|---|---|---|---|---|---|
| QFT | 20 | 50.3 | 38.7 | 15.1 | 9.4 | 13.4 |
| QFT | 24 | 2307 | 641 | 319 | 187 | 180 |
| Random, depth 100 | 20 | 55.2 | 30.5 | 8.5 | 6.8 | 10.9 |
| Random, depth 100 | 24 | 913 | 484 | 125 | 92 | 81 |

¹ Faster of Aer CPU with and without its fusion. Aer CPU timings were noisy on this laptop (in
one case the slowest of 7 runs took 30× the fastest), so read them as rough.
² Faster of Aer GPU with and without fusion.
³ cuStateVec driven gate by gate through its general `apply_matrix` call, with no fusion and none of
its specialised routines. This is a like-for-like kernel comparison, not cuStateVec's best possible use.

From 18 qubits up, qsim was 1.2–1.7× faster than this cuStateVec usage and 3.4–5.3× faster than
Aer GPU at 24 qubits. Up to 14 qubits, cuStateVec was 1.7–2× faster than qsim (0.47 vs 0.93 ms
for a 10-qubit QFT), since qsim pays more Python overhead per gate. The comparison covers one
laptop GPU and two circuit families. It isn't a general ranking of these simulators.

## Run it

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev,compare]" "cupy-cuda12x[ctk]"

.venv/bin/pytest -m "not gpu"        # CPU tests (run anywhere)
.venv/bin/pytest -m gpu              # GPU tests (NVIDIA GPU + CuPy)
.venv/bin/python -m bench.run_all    # every benchmark, results in bench/results/
```

Any Python from 3.12 works. A small example:

```python
from qsim.circuit import Circuit
from qsim.backends.cuda_backend import CudaBackend   # or: from qsim.backends import NumpyBackend

bell = Circuit(2).h(0).cx(0, 1)
sim = CudaBackend(2)
sim.run(bell)
print(sim.statevector())          # [0.707, 0, 0, 0.707]
print(sim.sample(1000, seed=1))   # only 0 (|00>) and 3 (|11>)
```

`notebooks/colab_run.ipynb` runs the tests and all benchmarks on a Colab GPU.

## Conventions

- Qubit 0 is the least significant bit of a state index (same as Qiskit).
- Pauli strings are written with the highest qubit first: `"XIZY"` means Y on qubit 0.
- Gate definitions follow Qiskit. Rz(θ) = diag(e^{−iθ/2}, e^{iθ/2}).
- GPU state is complex64 by default (complex128 optional). The reference is complex128.

## Limitations

- The comparison above covers one laptop GPU, two circuit families and cuStateVec's general
  per-gate API only. It doesn't show that qsim beats these libraries in general. Aer and
  cuStateVec do much more (noise, shots, batching, multi-GPU, many specialised routines).
- Shor's multiplication circuits are hard-coded for N = 15. It demonstrates the algorithm, not
  general factoring.
- H₂ in a minimal basis is exactly solvable classically, which is why it's the validation case.
  With two electrons, UCCSD can represent the exact ground state, so matching it shows correctness,
  not quantum advantage.
- Single GPU only. No noise models or density matrices.
- All benchmarks so far come from one 4 GB laptop GPU. The 29–30 qubit runs on a 16 GB GPU
  (Colab T4) are set up in the notebook but have not been run.

## Licence

MIT
