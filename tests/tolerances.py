"""Every numerical tolerance used by the tests, in one place.

COMPLEX128 values compare two double-precision calculations (oracle vs Aer, oracle vs itself),
so only rounding differences of ~1e-15 per operation are expected.
COMPLEX64 values compare the single-precision GPU backend with the double-precision oracle.
They were calibrated from measured errors; see docs/DECISIONS.md (tolerance calibration).
"""

# Double precision vs double precision
ATOL_COMPLEX128 = 1e-9

# Norm must stay 1 after every gate
NORM_ATOL_COMPLEX128 = 1e-10

# Single-precision GPU state vs double-precision oracle (max |amplitude difference|).
# Measured worst case over random circuits n=1..20, depth 100: 5.5e-7
# (bench/results/tolerance_complex64_*.json). 1e-5 keeps an ~18x safety margin.
ATOL_COMPLEX64 = 1e-5
NORM_ATOL_COMPLEX64 = 1e-5

# Expectation values computed on the GPU (accumulated in double) vs oracle
EXPECT_ATOL_COMPLEX64 = 1e-5
EXPECT_ATOL_COMPLEX128 = 1e-10

# Views on the GPU (marginal probabilities, Bloch vectors, step traces) vs the complex128
# oracle. Measured worst case: 2.4e-7 over 20 random 10-qubit traces, 8.2e-11 for marginals of
# 20-qubit states (bench/results/<gpu>/w2_steps.json, RTX 3050 Laptop). 1e-5 keeps a ~40x margin.
VIEW_ATOL_COMPLEX64 = 1e-5

# Chemical accuracy in Hartree: the VQE acceptance bound
CHEMICAL_ACCURACY = 1.6e-3
