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
| `qsimgpu/gates.py` | Exact gate matrices (Qiskit conventions) and which memory-access family each gate belongs to |
| `qsimgpu/circuit.py` | A circuit as plain data: validated `Gate` records, builder methods, `inverse()`, `depth()` |
| `qsimgpu/backends/numpy_backend.py` | The reference simulator: simple, double precision, used to judge everything else |
| `qsimgpu/backends/cuda_backend.py` + `qsimgpu/kernels/*.cu` | The GPU simulator: CUDA C kernels compiled at run time through CuPy |
| `qsimgpu/passes/fusion.py` | Gate fusion: merge consecutive gates into one matrix, so the state is streamed fewer times |
| `qsimgpu/algorithms/` | GHZ, Grover, QFT, Shor (N = 15), VQE for H₂ |
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

What the measurements show, in words (the numbers are in the generated tables below):

- **Memory bandwidth is the limit.** A general single-qubit gate runs at essentially the speed of a
  plain copy of the state, and from about 19 qubits on each extra qubit doubles the time per gate.
  Below that the state fits in the GPU's L2 cache and launch overhead dominates.
- **Moving fewer bytes is what helps.** Specialised kernels speed up exactly the gates where they
  read or write less memory (phase gates, CZ/CP, SWAP). Removing multiplications alone (X, CX, Rz)
  changes nothing.
- **Fusion helps at small widths only.** Merging gates on up to 2 qubits gives the best results;
  wider blocks need more arithmetic than the memory savings are worth on these GPUs. The first
  fusion kernel was slower than no fusion at all because its per-thread arrays didn't fit in
  registers; one compiled kernel per matrix size fixed it.
- **Double precision costs more than its bytes** on consumer GPUs, because their FP64 units are
  slow. complex64 stays the default; its error stays around 1e-7.
- **Comparison with other simulators** uses the same circuits, the same precision and the same
  returned value on every engine, with Aer's qubit truncation off. It covers one GPU, two circuit
  families and cuStateVec's general per-gate API only, so it isn't a general ranking.

The tables are generated from the JSON files by `python -m bench.report`. Each file also records
the GPU, driver, CUDA and library versions, live clocks and a bandwidth sanity check; files from an
unhealthy GPU are marked `"suspect"` and refused by the report. Results live in one folder per GPU
under `bench/results/`.

<!-- report:start -->
_Generated by `python -m bench.report` from `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/`. Do not edit by hand._

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), driver 535.309.01, CUDA runtime 12090, Python 3.14.8, commit `6f8587e`.

Copy bandwidth (B0): **168 GB/s**. The general single-qubit kernel on 24 qubits runs at **99.7–99.8% of a copy of the same state** (B3). Sources: `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b0_bandwidth.json`, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b3_efficiency_general.json`.

Time for one H gate vs number of qubits (B2, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b2_scaling_complex64.json`):

| Qubits | ms per gate | effective GB/s |
|---|---|---|
| 10 | 0.012 | 1 |
| 14 | 0.012 | 21 |
| 18 | 0.030 | 138 |
| 20 | 0.105 | 160 |
| 22 | 0.404 | 166 |
| 24 | 1.6 | 168 |
| 26 | 6.4 | 168 |
| 28 | 25.6 | 168 |

Specialised kernels vs the general kernel, 24 qubits, same run (B4, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b4_specialised.json`):

| Gate | General (ms) | Specialised (ms) | Speedup |
|---|---|---|---|
| h (general both ways) | 1.6 | 1.6 | 1.00× |
| x | 1.6 | 1.6 | 1.00× |
| z | 1.6 | 0.804 | 1.99× |
| t | 1.6 | 0.804 | 1.99× |
| rz | 1.6 | 1.6 | 1.00× |
| cx | 0.808 | 0.809 | 1.00× |
| cz | 0.807 | 0.406 | 1.99× |
| cp | 0.809 | 0.406 | 1.99× |
| ccx | 0.407 | 0.409 | 1.00× |
| swap | 1.6 | 0.805 | 2.01× |
| mcz (all other qubits control) | 0.017 | 0.013 | 1.32× |

Gate fusion, 22 qubits (B5, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b5_fusion.json`):

| Circuit | Variant | Passes | Time (ms) | vs general |
|---|---|---|---|---|
| qft | unfused, general kernels | 264 | 68.5 | 1.00× |
| qft | unfused, specialised kernels | 264 | 39.7 | 1.72× |
| qft | fused m=1 | 264 | 44.7 | 1.53× |
| qft | fused m=2 | 242 | 37.2 | 1.84× |
| qft | fused m=3 | 131 | 78.0 | 0.88× |
| qft | fused m=4 | 88 | 384 | 0.18× |
| qft | fused m=5 | 65 | 817 | 0.08× |
| random_d100 | unfused, general kernels | 100 | 32.4 | 1.00× |
| random_d100 | unfused, specialised kernels | 100 | 24.7 | 1.31× |
| random_d100 | fused m=1 | 99 | 24.7 | 1.31× |
| random_d100 | fused m=2 | 79 | 20.6 | 1.57× |
| random_d100 | fused m=3 | 55 | 22.3 | 1.45× |
| random_d100 | fused m=4 | 42 | 118 | 0.27× |
| random_d100 | fused m=5 | 31 | 327 | 0.10× |

Precision (B6, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b6_precision.json`). Largest complex64 error vs the complex128 reference: 1.1e-07.

| Qubits | complex64 (ms/gate) | complex128 (ms/gate) | Ratio |
|---|---|---|---|
| 20 | 0.108 | 0.269 | 2.49× |
| 22 | 0.407 | 1.1 | 2.77× |
| 24 | 1.6 | 4.6 | 2.84× |
| 26 | 6.4 | 18.0 | 2.82× |

Same circuits on other simulators, single precision, every engine returns the same ⟨Z⟩ value, full state simulated, median of 7 runs, times in ms (B7, `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b7_compare_*.json`; engines run: aer-cpu, aer-cpu-nofusion, aer-gpu, aer-gpu-nofusion, custatevec, qsim, qsim-fused):

| Circuit | Qubits | Aer CPU¹ | Aer GPU¹ | cuStateVec² | qsim | qsim + fusion |
|---|---|---|---|---|---|---|
| qft | 18 | 18.7 | 16.0 | 2.8 | 2.3 | 6.7 |
| qft | 20 | 50.3 | 38.7 | 15.1 | 9.4 | 13.4 |
| qft | 22 | 925 | 147 | 70.4 | 41.4 | 43.7 |
| qft | 24 | 2307 | 641 | 319 | 187 | 180 |
| random_d100 | 18 | 14.6 | 11.8 | 1.8 | 1.5 | 5.0 |
| random_d100 | 20 | 55.2 | 30.5 | 8.5 | 6.7 | 10.9 |
| random_d100 | 22 | 254 | 114 | 32.8 | 26.3 | 26.6 |
| random_d100 | 24 | 913 | 484 | 125 | 91.9 | 81.3 |

¹ Faster of the runs with and without Aer's gate fusion. ² cuStateVec driven gate by gate through its general `apply_matrix` call.

Capacity (B8):

- complex64: **28 qubits** verified by norm and sampling; 29 qubits ran out of memory (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b8_scale_complex64.json`).
- complex128: **27 qubits** verified by norm and sampling; 28 qubits ran out of memory (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/b8_scale_complex128.json`).

H₂ VQE (chemical accuracy is 1.6e-3 Ha):

- NumPy, complex128: worst |VQE − exact| over 45 bond lengths **1.4e-10 Ha**, minimum at 0.75 Å (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/vqe_h2_4q_numpy.json`).
- GPU, complex64: worst |VQE − exact| over 45 bond lengths **3.3e-05 Ha**, minimum at 0.75 Å (`bench/results/nvidia-geforce-rtx-3050-laptop-gpu/vqe_h2_4q_cuda.json`).

<!-- report:end -->

## Run it

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev,compare]" "cupy-cuda12x[ctk]"

.venv/bin/pytest -m "not gpu"        # CPU tests (run anywhere)
.venv/bin/pytest -m gpu              # GPU tests (NVIDIA GPU + CuPy)
.venv/bin/python -m bench.run_all    # every benchmark, results in bench/results/
```

Any Python from 3.12 works. The package installs as `qsim-gpu` and is imported as `qsimgpu`
(the shorter name `qsim` is already taken on PyPI). A small example:

```python
from qsimgpu.circuit import Circuit
from qsimgpu.backends.cuda_backend import CudaBackend   # or: from qsimgpu.backends import NumpyBackend

bell = Circuit(2).h(0).cx(0, 1)
sim = CudaBackend(2)
sim.run(bell)
print(sim.statevector())          # [0.707, 0, 0, 0.707]
print(sim.sample(1000, seed=1))   # only 0 (|00>) and 3 (|11>)
```

`scripts/gpu_cloud_run.sh` runs the tests and all benchmarks on a cloud GPU: on a free Kaggle T4
with `python scripts/kaggle_run.py push`, or on Colab with `notebooks/colab_run.ipynb`.
`python -m bench.report` then rewrites the result tables in this README from the JSON files.

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
