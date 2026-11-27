"""The 3-qubit bit-flip code: protect one qubit against a single X error.

Qubits 0-2 hold the code word a|000> + b|111>; qubits 3 and 4 are ancillas.
  encode:   CX 0->1, CX 0->2
  error:    optionally X on one of qubits 0-2
  syndrome: ancilla 3 = parity of qubits 0,1; ancilla 4 = parity of qubits 1,2; measured into
            clbits 0 and 1. Register value (bit j = clbit j): 1 -> qubit 0 flipped,
            3 -> qubit 1 flipped, 2 -> qubit 2 flipped, 0 -> no error.
  correct:  X on the indicated qubit, conditioned on the register value
  decode:   CX 0->2, CX 0->1, leaving the protected state on qubit 0
The syndrome reveals where the error is without revealing (or disturbing) a and b.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qsimgpu.algorithms.teleport import preparation_matrix
from qsimgpu.backends.base import Backend
from qsimgpu.circuit import Circuit
from qsimgpu.execution import RngLike

SYNDROME_TO_QUBIT = {1: 0, 3: 1, 2: 2}


def bitflip_circuit(psi: NDArray[np.complexfloating], error_qubit: int | None) -> Circuit:
    c = Circuit(5, n_clbits=2)
    c.unitary(preparation_matrix(psi), [0])
    c.cx(0, 1).cx(0, 2)                                   # encode
    if error_qubit is not None:
        if error_qubit not in (0, 1, 2):
            raise ValueError("error_qubit must be 0, 1, 2 or None")
        c.x(error_qubit)                                  # the error
    c.cx(0, 3).cx(1, 3).cx(1, 4).cx(2, 4)                 # syndrome extraction
    c.measure(3, 0).measure(4, 1)
    for value, q in SYNDROME_TO_QUBIT.items():            # correction
        c.x(q).c_if([0, 1], value)
    c.cx(0, 2).cx(0, 1)                                   # decode
    return c


def encode_flip_correct(backend: Backend, psi: NDArray[np.complexfloating],
                        error_qubit: int | None, rng: RngLike = None) -> NDArray[np.complex128]:
    """Run the code and return the recovered single-qubit state of qubit 0."""
    backend.reset()
    s0, s1 = backend.run(bitflip_circuit(psi, error_qubit), rng=rng)
    sv = backend.statevector()
    # after decoding, qubits 1-2 are |00> and the ancillas hold the syndrome bits
    base = (s0 << 3) | (s1 << 4)
    return np.array([sv[base], sv[base | 1]], dtype=np.complex128)
