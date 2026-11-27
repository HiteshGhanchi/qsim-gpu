# qsim-gpu: one-page summary

**What.** A GPU statevector quantum circuit simulator written from scratch (Python + CUDA C via
CuPy), checked against a NumPy reference and Qiskit Aer, running Grover, QFT, Shor for N = 15
and VQE for the H₂ molecule.

**Why.** Simulating n qubits takes 2ⁿ complex numbers, so each qubit doubles memory and time.
The project asks where a GPU simulator spends its time and how close it gets to the hardware's
memory-bandwidth limit.

**Key insight.** A gate is one streaming pass over the state with about one flop per byte, so
the simulator is memory-bound. The minimum time per gate is 16 · 2ⁿ bytes (complex64) divided by
achievable bandwidth, measured with a copy kernel.

**How it works.** A circuit is plain data (validated `Gate` records). Two backends implement
one interface: a double-precision NumPy reference that applies every gate with one tensor
contraction, and a CUDA backend where each thread owns one independent group of amplitudes (the
"insert a zero bit" index trick, 64-bit indices, no synchronisation). Measurement uses two-pass
chunked sampling, so it never needs a second full-size array. Pauli expectations are one pass
using bit masks, summed in double.

**Correctness.** The NumPy reference matches Aer exactly on random circuits (no reordering or
phase fixes), and the GPU matches NumPy on random circuits up to 20 qubits. Every gate is checked
against explicit matrices. Norm and inverse invariants hold. Algorithms give their known answers.
VQE matches exact diagonalisation of H₂ at 45 bond lengths (within 1.4e-10 Ha on CPU and 3.3e-5 Ha on
GPU in single precision, against a 1.6e-3 Ha requirement). The single-precision tolerance came from
a measurement (worst error 5.5e-7).

**Optimisations, each measured before/after.**
0. Baseline: the general single-qubit kernel already runs at 99.5–99.8% of measured copy
   bandwidth (168 GB/s on an RTX 3050 Laptop GPU).
1. Specialised kernels per gate family. Cutting arithmetic gained nothing; cutting bytes gained
   1.99× (phase gates, CZ, CP) and 2.01× (SWAP). That's the proof of memory-boundness.
2. Gate fusion. The first version was slower despite fewer passes, because of register spills.
   With one compiled kernel per matrix size, width-2 fusion ran the QFT 1.84× and a random
   circuit 1.57× faster than the baseline.
3. Precision: complex128 costs 2.5–2.8× per gate (more than the 2× byte ratio, because consumer
   FP64 is slow) and one qubit of capacity.

**Fair comparison** (same circuits, same single precision, same returned value, full state on every
engine): from 18 qubits up qsim was 1.2–1.7× faster than cuStateVec driven through its general
per-gate API, and 3.4–5.3× faster than Qiskit Aer GPU at 24 qubits. Below 14 qubits cuStateVec was faster.

**Scale.** 28 qubits in complex64 and 27 in complex128 on a 4 GB laptop GPU, verified by norm and
sampling.

**Honest limits.** One 4 GB laptop GPU, two circuit families, and cuStateVec's general API only,
so this is not a general ranking of simulators. The 29–30 qubit T4 runs are still to do.
Shor's arithmetic is hard-coded for 15. H₂ is a validation case, not a quantum advantage.
