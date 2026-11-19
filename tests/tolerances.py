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

# Chemical accuracy in Hartree: the VQE acceptance bound
CHEMICAL_ACCURACY = 1.6e-3
