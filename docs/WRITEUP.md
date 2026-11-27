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
VQE matches exact diagonalisation of H₂ at 45 bond lengths (within 1.4e-10 Ha on CPU and
7.3e-5 Ha on the T4 in single precision, against a 1.6e-3 Ha requirement). The single-precision tolerance came from
a measurement (worst error 5.5e-7).

**Optimisations, each measured before/after** (headline GPU: 16 GB Tesla T4 on Kaggle; files in
`bench/results/tesla-t4/`, laptop results in `bench/results/nvidia-geforce-rtx-3050-laptop-gpu/`).
0. Baseline: a general single-qubit gate on 26 qubits moves data at 234 GB/s, the same as a plain
   copy of the state measured in the same run (230 GB/s) and 73% of the T4's published 320 GB/s
   peak. On the laptop GPU: 99.5–99.8% of copy speed.
1. Specialised kernels per gate family. Cutting arithmetic gained nothing; cutting bytes gained
   about 2× (T4: 2.07–2.08× for phase gates, CZ, CP; 1.98× for SWAP). That's the proof of
   memory-boundness.
2. Gate fusion. The first version was slower despite fewer passes, because of register spills.
   With one compiled kernel per matrix size, fusing up to 2–3 qubits ran the QFT 1.92× and a
   random circuit 1.75× faster than the general-kernel baseline on the T4 (24 qubits).
3. Precision: complex128 costs one qubit of capacity and more than the 2× byte ratio in time
   (laptop 2.5–2.8×; T4 3.6–3.9× at 20–24 qubits but 2.0× at 26, still to be re-checked).

**Fair comparison** (T4, same circuits, single precision, same returned value, full state on every
engine): from 20 qubits up qsim was 1.3–2.2× faster than cuStateVec driven through its general
per-gate API and 2.2–7.0× faster than Qiskit Aer GPU. At 18 qubits and below cuStateVec was faster.

**Scale.** 30 qubits in complex64 and 29 in complex128 on the 16 GB T4 (28 and 27 on the 4 GB
laptop GPU), each verified by norm and sampling.

**Honest limits.** Two GPUs, two circuit families, and cuStateVec's general API only, so this is
not a general ranking of simulators.
Shor's arithmetic is hard-coded for 15. H₂ is a validation case, not a quantum advantage.
