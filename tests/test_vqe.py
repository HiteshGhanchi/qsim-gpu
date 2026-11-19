"""VQE for H2: building blocks, Hamiltonian data, and chemical accuracy over the grid."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from qsim.algorithms import vqe
from qsim.backends import NumpyBackend
from qsim.circuit import Circuit
from tests.circuits import random_pauli, random_state
from tests.tolerances import CHEMICAL_ACCURACY

DATA = vqe.load_h2_data()


def _apply(c: Circuit, psi: np.ndarray) -> np.ndarray:
    b = NumpyBackend(c.n_qubits)
    b.set_state(psi)
    b.run(c)
    return b.statevector()


@pytest.mark.parametrize("seed", range(20))
def test_pauli_rotation_matches_matrix_exponential(seed: int) -> None:
    n = 1 + seed % 4
    pauli = random_pauli(n, seed)
    phi = 0.37 * (seed + 1)
    c = Circuit(n)
    vqe.add_pauli_rotation(c, pauli, phi)
    psi = random_state(n, 500 + seed)
    expected = expm(1j * phi * vqe.pauli_matrix(pauli)) @ psi
    if set(pauli) == {"I"}:
        expected = psi  # exp(i*phi*I) is a global phase; the circuit drops it
    np.testing.assert_allclose(_apply(c, psi), expected, atol=1e-12)


def test_generator_terms_commute_so_one_trotter_step_is_exact() -> None:
    for gen in vqe.uccsd_generators():
        mats = [vqe.pauli_matrix(p) for _, p in gen]
        for a in mats:
            for b in mats:
                np.testing.assert_allclose(a @ b, b @ a, atol=1e-12)


@pytest.mark.parametrize("k", range(3))
def test_each_excitation_circuit_equals_exp_of_generator(k: int) -> None:
    theta = 0.3
    gen = vqe.uccsd_generators()[k]
    g_matrix = sum(1j * g * vqe.pauli_matrix(p) for g, p in gen)
    hf = np.zeros(16, dtype=complex)
    hf[vqe.HF_STATE_4Q] = 1
    c = Circuit(4)
    for g, p in gen:
        vqe.add_pauli_rotation(c, p, theta * g)
    np.testing.assert_allclose(_apply(c, hf), expm(theta * g_matrix) @ hf, atol=1e-12)


def test_double_excitation_moves_hf_to_doubly_excited_state() -> None:
    sv = _apply(vqe.uccsd_circuit([0, 0, np.pi / 2]), np.eye(16)[0])
    assert abs(sv[0b1100]) == pytest.approx(1, abs=1e-12)


def test_hamiltonian_data_cross_checks() -> None:
    assert len(DATA) == 45
    assert DATA[0]["bond_length_angstrom"] == 0.3 and DATA[-1]["bond_length_angstrom"] == 2.5
    for p in DATA:
        e4 = vqe.exact_ground_energy(p["terms_4q"], n_electrons=2)
        e2 = vqe.exact_ground_energy(p["terms_2q"])
        assert e4 == pytest.approx(p["fci_energy"], abs=1e-8)
        assert e2 == pytest.approx(p["fci_energy"], abs=1e-8)
        hf = vqe.hamiltonian_matrix(p["terms_4q"])[vqe.HF_STATE_4Q, vqe.HF_STATE_4Q].real
        assert hf == pytest.approx(p["hf_energy"], abs=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["2q", "4q"])
def test_vqe_reaches_chemical_accuracy_on_whole_grid(stage: str) -> None:
    scan = vqe.run_scan(NumpyBackend, stage)
    errors = [pt.error for pt in scan.points]
    assert max(abs(e) for e in errors) < CHEMICAL_ACCURACY
    assert min(errors) > -1e-9  # variational: never below the exact energy
    best = min(scan.points, key=lambda pt: pt.energy)
    assert abs(best.bond_length - 0.74) <= 0.05
    assert best.energy == pytest.approx(best.fci, abs=CHEMICAL_ACCURACY)
    if stage == "4q":
        singles = [abs(t) for pt in scan.points for t in pt.theta[:2]]
        assert max(singles) < 1e-3  # singles stay ~0 by symmetry


@pytest.mark.gpu
def test_vqe_on_gpu_selected_points() -> None:
    from qsim.backends.cuda_backend import CudaBackend

    pts = [p for p in DATA if p["bond_length_angstrom"] in (0.3, 0.75, 1.5, 2.5)]
    scan = vqe.run_scan(lambda n: CudaBackend(n), "4q", points=pts, fd_step=1e-4)
    for pt in scan.points:
        assert abs(pt.error) < CHEMICAL_ACCURACY
