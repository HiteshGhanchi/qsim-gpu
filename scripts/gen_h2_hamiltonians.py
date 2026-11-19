"""Offline: generate qubit Hamiltonians for H2 (STO-3G) over a grid of bond lengths.

Needs pyscf and openfermion (not runtime dependencies). Run once and commit the JSON:
    .venv-chem/bin/python scripts/gen_h2_hamiltonians.py

For each bond length R:
  1. PySCF: restricted Hartree-Fock, then full CI (the exact answer in this basis).
  2. Molecular-orbital integrals -> second-quantised Hamiltonian (OpenFermion).
  3. Jordan-Wigner -> 4-qubit Pauli Hamiltonian.
  4. Symmetry-conserving Bravyi-Kitaev with 2 electrons -> 2-qubit reduced Hamiltonian.
Coefficients come only from these calculations, never typed by hand.

Pauli strings are written leftmost = highest qubit, the convention used by qsim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import symmetry_conserving_bravyi_kitaev
from pyscf import ao2mo, fci, gto, scf

OUT = Path(__file__).resolve().parent.parent / "data" / "h2_hamiltonians.json"
GRID = [round(0.30 + 0.05 * i, 2) for i in range(45)]  # 0.30 .. 2.50 Angstrom


def qubit_terms(qubit_op, n_qubits: int) -> list[list[object]]:  # type: ignore[no-untyped-def]
    terms = []
    for ops, coeff in sorted(qubit_op.terms.items()):
        if abs(coeff) < 1e-12:
            continue
        assert abs(complex(coeff).imag) < 1e-12, "Hamiltonian coefficients must be real"
        chars = ["I"] * n_qubits
        for q, p in ops:
            chars[n_qubits - 1 - q] = p
        terms.append([float(complex(coeff).real), "".join(chars)])
    return terms


def one_geometry(r: float) -> dict[str, object]:
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    e_hf = float(mf.kernel())
    e_fci = float(fci.FCI(mf).kernel()[0])
    c = mf.mo_coeff
    n_orb = c.shape[1]
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), n_orb)        # chemist order (pq|rs)
    two_body = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # OpenFermion's order
    one_sp, two_sp = spinorb_from_spatial(h1, two_body)
    e_nuc = float(mol.energy_nuc())
    ham = InteractionOperator(e_nuc, one_sp, 0.5 * two_sp)
    fermion = get_fermion_operator(ham)

    jw = jordan_wigner(fermion)
    reduced = symmetry_conserving_bravyi_kitaev(fermion, active_orbitals=4, active_fermions=2)

    # Self-checks before writing anything.
    e_red = float(np.linalg.eigvalsh(get_sparse_operator(reduced).toarray()).min())
    assert abs(e_red - e_fci) < 1e-8, (r, e_red, e_fci)
    return {
        "bond_length_angstrom": r,
        "nuclear_repulsion": e_nuc,
        "hf_energy": e_hf,
        "fci_energy": e_fci,
        "n_electrons": 2,
        "terms_4q": qubit_terms(jw, 4),
        "terms_2q": qubit_terms(reduced, 2),
    }


def main() -> None:
    import openfermion
    import pyscf

    points = []
    for r in GRID:
        points.append(one_geometry(r))
        print(f"R={r:.2f}  HF={points[-1]['hf_energy']:.6f}  FCI={points[-1]['fci_energy']:.6f}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "molecule": "H2", "basis": "sto-3g", "units": {"length": "angstrom", "energy": "hartree"},
        "mapping_4q": "jordan_wigner (spin orbitals interleaved: even = alpha, odd = beta)",
        "mapping_2q": "symmetry_conserving_bravyi_kitaev, 4 active orbitals, 2 electrons",
        "pauli_order": "leftmost character = highest qubit index",
        "generated_with": {"pyscf": pyscf.__version__, "openfermion": openfermion.__version__},
        "points": points,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
