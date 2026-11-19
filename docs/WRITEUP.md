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
VQE matches exact diagonalisation of H₂ at 45 bond lengths (within 1.4e-10 Ha on CPU and 6.3e-5 Ha on
GPU in single precision, against a 1.6e-3 Ha requirement). The single-precision tolerance came from
a measurement (worst error 5.5e-7).

**Optimisations, each measured before/after.**
1. Specialised kernels per gate family. Cutting arithmetic gained nothing; cutting bytes gained
   2.03× (phase gates, CZ, CP) and 3.57× (SWAP). That's the proof of memory-boundness.
2. Gate fusion. The first version was slower despite fewer passes, because of register spills.
   With one compiled kernel per matrix size, width-3 fusion ran a random circuit 2.08× faster than
   the baseline.
3. Precision: complex128 costs exactly 2.0× per gate (the byte ratio) and one qubit of capacity.

**Scale.** 28 qubits in complex64 and 27 in complex128 on a 4 GB laptop GPU, verified by norm and
sampling.

**Honest limits.** Not faster than Aer or cuStateVec, and not meant to be. Benchmarks so far ran on a
laptop GPU in a faulty clock state, so only within-run ratios are reported, and a clean re-run
is pending. Shor's arithmetic is hard-coded for 15. H₂ is a validation case, not a quantum
advantage.
