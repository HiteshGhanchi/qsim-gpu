"""VQE for H2 in the STO-3G basis.

The variational quantum eigensolver: a parameterised circuit prepares a trial state, the
simulator returns its exact energy <psi(theta)|H|psi(theta)>, and a classical optimiser
(SciPy) adjusts theta to minimise it. The lowest energy reached approximates the ground state.

Two stages:
- 2 qubits: the symmetry-reduced Hamiltonian with a small Ry/CX ansatz (plumbing check).
- 4 qubits: the Jordan-Wigner Hamiltonian with a UCCSD ansatz: start from the Hartree-Fock
  state |0011> (spin orbitals 0 and 1 occupied), then apply exp(theta * (T - T^dagger)) for two
  single excitations (0->2, 1->3) and one double excitation (0,1 -> 2,3).

The excitation generators are built here in NumPy from Jordan-Wigner fermion operators and
split into Pauli strings numerically, so no chemistry library is needed at run time. Each
exp(i*phi*P) becomes a basis change, a CX ladder, one Rz, and the undo.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qsim.backends.base import Backend
from qsim.circuit import Circuit

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "h2_hamiltonians.json"
Terms = list[tuple[float, str]]

_P1 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# --- data and exact reference -------------------------------------------------------

def load_h2_data(path: Path = DATA_FILE) -> list[dict[str, Any]]:
    """Points of data/h2_hamiltonians.json, with terms as (coeff, pauli) tuples."""
    points: list[dict[str, Any]] = json.loads(path.read_text())["points"]
    for p in points:
        p["terms_4q"] = [(float(c), str(s)) for c, s in p["terms_4q"]]
        p["terms_2q"] = [(float(c), str(s)) for c, s in p["terms_2q"]]
    return points


def pauli_matrix(pauli: str) -> NDArray[np.complex128]:
    """Dense matrix of a Pauli string (leftmost char = highest qubit = first Kronecker factor)."""
    m = np.array([[1.0 + 0j]])
    for ch in pauli:
        m = np.kron(m, _P1[ch])
    return m


def hamiltonian_matrix(terms: Terms) -> NDArray[np.complex128]:
    n = len(terms[0][1])
    h = np.zeros((2**n, 2**n), dtype=complex)
    for c, p in terms:
        h += c * pauli_matrix(p)
    return h


def exact_ground_energy(terms: Terms, n_electrons: int | None = None) -> float:
    """Lowest eigenvalue, optionally only among basis states with n_electrons ones (JW)."""
    h = hamiltonian_matrix(terms)
    if n_electrons is not None:
        keep = [i for i in range(h.shape[0]) if bin(i).count("1") == n_electrons]
        h = h[np.ix_(keep, keep)]
    return float(np.linalg.eigvalsh(h).min())


# --- building blocks -----------------------------------------------------------------

def add_pauli_rotation(c: Circuit, pauli: str, phi: float) -> None:
    """Append exp(i * phi * P) for a Pauli string P (leftmost char = highest qubit).

    Rotate each X/Y qubit into the Z basis, gather the parity onto the lowest active qubit
    with a CX ladder, apply Rz(-2*phi) there (exp(i*phi*Z) = Rz(-2*phi)), then undo.
    """
    n = len(pauli)
    active = [n - 1 - pos for pos, ch in enumerate(pauli) if ch != "I"]
    if not active:
        return  # identity: only a global phase
    active.sort()
    for q in active:
        ch = pauli[n - 1 - q]
        if ch == "X":
            c.h(q)
        elif ch == "Y":
            c.rx(math.pi / 2, q)
    for a, b in zip(active[::-1], active[-2::-1], strict=False):
        c.cx(a, b)
    c.rz(-2 * phi, active[0])
    for a, b in reversed(list(zip(active[::-1], active[-2::-1], strict=False))):
        c.cx(a, b)
    for q in active:
        ch = pauli[n - 1 - q]
        if ch == "X":
            c.h(q)
        elif ch == "Y":
            c.rx(-math.pi / 2, q)


def _annihilation(j: int, n: int) -> NDArray[np.complex128]:
    """Jordan-Wigner a_j on n qubits: Z on qubits below j, (X + iY)/2 on qubit j."""
    lower = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1| removes a fermion from qubit j
    m = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):  # highest qubit is the first Kronecker factor
        m = np.kron(m, lower if q == j else (_P1["Z"] if q < j else _P1["I"]))
    return m


def pauli_decomposition(matrix: NDArray[np.complex128], tol: float = 1e-12
                        ) -> list[tuple[complex, str]]:
    """Write a 2^n x 2^n matrix as sum c_P * P. c_P = Tr(P M) / 2^n."""
    n = int(round(math.log2(matrix.shape[0])))
    out = []
    for chars in product("IXYZ", repeat=n):
        p = "".join(chars)
        coeff = complex(np.trace(pauli_matrix(p) @ matrix) / 2**n)
        if abs(coeff) > tol:
            out.append((coeff, p))
    return out


@cache
def uccsd_generators(n_qubits: int = 4) -> tuple[tuple[tuple[float, str], ...], ...]:
    """Pauli form of the three H2 excitation generators T - T^dagger.

    Each generator G is anti-Hermitian, so G = i * sum g_P P with real g_P. Returned as
    ((g_P, P), ...) per excitation, in the order: single 0->2, single 1->3, double 01->23.
    """
    a = [_annihilation(j, n_qubits) for j in range(n_qubits)]
    ad = [m.conj().T for m in a]
    excitations = [ad[2] @ a[0], ad[3] @ a[1], ad[3] @ ad[2] @ a[1] @ a[0]]
    result = []
    for t in excitations:
        g = t - t.conj().T
        terms = []
        for coeff, p in pauli_decomposition(g):
            assert abs(coeff.real) < 1e-12, "generator must be anti-Hermitian"
            terms.append((float(coeff.imag), p))
        result.append(tuple(terms))
    return tuple(result)


# --- ansatz circuits -------------------------------------------------------------------

HF_STATE_4Q = 0b0011  # spin orbitals 0 (alpha) and 1 (beta) occupied


def uccsd_circuit(theta: Sequence[float]) -> Circuit:
    """HF state, then exp(theta_k * G_k) for the three H2 excitations (one Trotter step)."""
    c = Circuit(4).x(0).x(1)
    for t, gen in zip(theta, uccsd_generators(), strict=True):
        for g, p in gen:
            add_pauli_rotation(c, p, t * g)
    return c


def two_qubit_ansatz(theta: Sequence[float]) -> Circuit:
    """Ry layer, CX, Ry layer: reaches any real 2-qubit state (used with the reduced H)."""
    t0, t1, t2, t3 = theta
    return Circuit(2).ry(t0, 0).ry(t1, 1).cx(0, 1).ry(t2, 0).ry(t3, 1)


# --- the optimisation loop ---------------------------------------------------------------

@dataclass
class VQEPoint:
    bond_length: float
    energy: float
    exact: float
    fci: float
    hf: float
    theta: list[float]
    evaluations: int

    @property
    def error(self) -> float:
        return self.energy - self.exact


@dataclass
class VQEScan:
    stage: str
    points: list[VQEPoint] = field(default_factory=list)


Ansatz = Callable[[Sequence[float]], Circuit]


def energy_function(backend: Backend, terms: Terms,
                    ansatz: Ansatz) -> Callable[[NDArray[Any]], float]:
    def energy(theta: NDArray[Any]) -> float:
        backend.reset()
        backend.run(ansatz([float(t) for t in theta]))
        return backend.expectation_hamiltonian(terms)
    return energy


def minimise(energy: Callable[[NDArray[Any]], float], theta0: NDArray[Any],
             fd_step: float = 1e-5) -> tuple[NDArray[Any], float, int]:
    from scipy.optimize import minimize

    res = minimize(energy, theta0, method="L-BFGS-B",
                   options={"eps": fd_step, "ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    return np.asarray(res.x), float(res.fun), int(res.nfev)


def run_scan(make_backend: Callable[[int], Backend], stage: str = "4q",
             points: list[dict[str, Any]] | None = None, fd_step: float = 1e-5) -> VQEScan:
    """VQE over the bond-length grid, warm-starting each point from the previous optimum."""
    data = load_h2_data() if points is None else points
    if stage == "4q":
        n, key, ansatz, theta = 4, "terms_4q", uccsd_circuit, np.zeros(3)
    elif stage == "2q":
        n, key, ansatz, theta = 2, "terms_2q", two_qubit_ansatz, np.array([np.pi, 0, 0, 0])
    else:
        raise ValueError("stage must be '2q' or '4q'")
    backend = make_backend(n)
    scan = VQEScan(stage)
    for p in data:
        terms = p[key]
        exact = exact_ground_energy(terms, n_electrons=2 if stage == "4q" else None)
        theta, e, nfev = minimise(energy_function(backend, terms, ansatz), theta, fd_step)
        scan.points.append(VQEPoint(p["bond_length_angstrom"], e, exact, p["fci_energy"],
                                    p["hf_energy"], [float(t) for t in theta], nfev))
    return scan
