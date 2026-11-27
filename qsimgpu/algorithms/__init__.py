"""Algorithms built only on the Circuit IR and the Backend interface (never on CuPy directly).

Submodules: grover, qft, shor15, teleport, bitflip, vqe (vqe's optimiser needs scipy).
"""

from qsimgpu.algorithms import bitflip, grover, qft, shor15, teleport, vqe

__all__ = ["bitflip", "grover", "qft", "shor15", "teleport", "vqe"]
