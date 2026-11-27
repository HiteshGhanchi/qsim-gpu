"""qsim: a statevector quantum circuit simulator with a NumPy reference and a CUDA backend.

Conventions used everywhere in this package:
- Little-endian qubit order: qubit 0 is the least significant bit of a state index.
- Gate definitions follow Qiskit.
- The CUDA backend defaults to complex64; the NumPy reference uses complex128.
"""

__version__ = "0.1.0"
