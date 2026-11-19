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

## Results so far

All numbers come from JSON files in `bench/results/`.

> **Important caveat.** The benchmarks were run on a 4 GB RTX 3050 Laptop GPU whose driver was
> in a faulty clock state during the session ("System is not in ready state"; DRAM bandwidth
> measured at only ~8 GB/s). Absolute times are therefore far slower than this GPU can do and
> are **not** quoted as performance figures. Ratios measured inside one run (same GPU state for
> both sides) are meaningful and are what's reported below. A clean re-run (`python -m bench.run_all`)
> after a reboot, and on a Colab T4, is still to do.

**It is memory-bound** (B2, B3). From 18 qubits up, each extra qubit doubles the time per
gate. Below that the state fits in the 2 MB L2 cache and launch overhead dominates. A general
single-qubit gate runs at 99.6–100% of the speed of a plain copy of the same state.

**Moving fewer bytes is what helps** (B4, 24 qubits). The specialised kernels:

| Gate | Speedup over the general kernel | Reason |
|---|---|---|
| Z, T | 2.03× | only amplitudes with the target bit set are read |
| CZ, CP | 2.03× | only the \|11⟩ quarter is touched |
| SWAP | 3.57× | only \|01⟩ and \|10⟩ move |
| X, CX, Rz | 1.00–1.01× | fewer multiplications but the same bytes: no gain |

**Gate fusion** (B5, 22 qubits). Merging gates on up to 3 qubits cut a depth-100 random
circuit from 100 passes to 55 and ran it 2.08× faster than the baseline. The first version of
fusion was *slower* despite fewer passes. Its kernel kept per-thread arrays too large for
registers, which was fixed by compiling one kernel per matrix size. Fusing 4–5 qubits is slower
on this GPU, so the default width is 3. For the QFT, width 2 is best, because its many controlled-phase
gates are already cheap with the specialised kernel.

**Precision** (B6). complex128 takes exactly 2.0× as long as complex64 per gate, which is the
ratio of their sizes, and costs one qubit of capacity. complex64's error stays around 1e-7, so
it's the default.

**Capacity** (B8). On the 4 GB GPU: 28 qubits in complex64 and 27 in complex128, each verified by
norm and sampling. 29 qubits (4 GB) runs out of memory.

**VQE for H₂.** Across 0.30–2.50 Å the VQE energy stays within 1.4e-10 Hartree of exact
diagonalisation on the NumPy backend, and within 6.3e-5 Hartree on the GPU in single precision.
Chemical accuracy is 1.6e-3. The minimum is at 0.75 Å, −1.137117 Hartree.

**Compared with other simulators** (B7, same circuits, same degraded GPU). Up to 14 qubits, qsim and NVIDIA
cuStateVec finish a circuit in about 1 ms, and Qiskit Aer takes 3–13 ms (setup cost per run).
From 18 qubits up, cuStateVec is 5–14% faster than qsim on these unfused circuits (that qsim run
used the general kernels, before the specialised ones were added).
Aer GPU is the fastest at 24 qubits, most likely because it fuses gates by default. These
comparisons need repeating on a healthy GPU before any claim is made either way.

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

- Not faster than Qiskit Aer or cuStateVec, and not trying to be. The aim was to understand and
  measure the gap.
- Shor's multiplication circuits are hard-coded for N = 15. It demonstrates the algorithm, not
  general factoring.
- H₂ in a minimal basis is exactly solvable classically, which is why it's the validation case.
  With two electrons, UCCSD can represent the exact ground state, so matching it shows correctness,
  not quantum advantage.
- Single GPU only. No noise models or density matrices.
- Benchmarks so far come from one laptop GPU in a degraded clock state (see the caveat above).

## Licence

MIT
