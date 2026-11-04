"""Hold the tabulated H2/STO-3G record to a first-principles recomputation.

The scratch engine lives in ``h2_sto3g_reference.py`` next to this file: it
evaluates the three-Gaussian STO-3G contraction analytically, with no PySCF and
no SuperFermion call, so these assertions compare the record against the basis
set definition rather than against itself.  The audit harness ``_xq4_chem.py``
loads that same module by path, so the numbers reported there and the numbers
asserted here cannot drift apart.

The engine is kept out of this file on purpose.  The audit replay runs from the
repository root and therefore does not load ``tests/conftest.py``, which is
where the ``domain`` marker is registered; a module that merely imports the
engine must not carry ``pytestmark`` or it emits an unknown-mark warning when
the harness loads it.
"""

import numpy as np
import pytest
from h2_sto3g_reference import (
    BOHR_PER_ANGSTROM,
    RECORD_D,
    bisect,
    energies,
    fci_from,
    golden_min,
    h2_integrals,
    hf_from,
)

from superfermion.chemistry.library import MoleculeLibrary

pytestmark = pytest.mark.domain


@pytest.fixture(scope="module")
def record() -> dict:
    return MoleculeLibrary.require_by_name("H2")


@pytest.fixture(scope="module")
def analytic() -> dict:
    return h2_integrals(RECORD_D)


def test_contracted_basis_functions_are_normalised(analytic):
    """A mis-normalised contraction would shift every integral by a constant."""
    s, C = analytic["s"], analytic["C"]
    assert np.max(np.abs(np.diag(s) - 1.0)) < 1e-12
    assert np.max(np.abs(C.T @ s @ C - np.eye(2))) < 1e-12
    assert 0.0 < s[0, 1] < 1.0


def test_record_integrals_match_the_analytic_ones(record, analytic):
    """Every stored number, against an independent evaluation of the basis."""
    h1, h2 = analytic["h1"], analytic["h2"]
    d1 = float(np.max(np.abs(h1 - np.asarray(record["h1"], dtype=float))))
    d2 = float(np.max(np.abs(h2 - np.asarray(record["h2"], dtype=float))))
    assert d1 < 5e-11, f"h1 differs by {d1:.3e}: {h1.tolist()} vs {record['h1']}"
    assert d2 < 5e-11, f"h2 differs by {d2:.3e}"


def test_record_nuclear_repulsion_follows_from_its_geometry(record):
    """``1/R`` in atomic units, from the geometry string the record stores."""
    xyz = record["geometry"]
    z = float(xyz.split(";")[1].split()[3])
    r_bohr = z * BOHR_PER_ANGSTROM
    assert abs(float(record["nuclear_repulsion"]) - 1.0 / r_bohr) < 1e-10


def test_record_hf_energy_follows_from_its_own_integrals(record):
    """The stored ``hf_energy`` is the RHF expectation of the stored integrals."""
    h1 = np.asarray(record["h1"], dtype=float)
    h2 = np.asarray(record["h2"], dtype=float)
    got = hf_from(h1, h2, float(record["nuclear_repulsion"]))
    assert abs(got - float(record["hf_energy"])) < 1e-11


def test_record_fci_energy_follows_from_its_own_integrals(record):
    """The stored ``fci_energy`` is the exact 2x2 CI of the stored integrals.

    This is the row that catches the Coulomb-for-exchange mistake: feeding
    ``(00|11)`` into the off-diagonal instead of ``(01|01)`` gives -1.3579875.
    """
    h1 = np.asarray(record["h1"], dtype=float)
    h2 = np.asarray(record["h2"], dtype=float)
    got = fci_from(h1, h2, float(record["nuclear_repulsion"]))
    assert abs(got - float(record["fci_energy"])) < 1e-11
    assert got < float(record["hf_energy"]), "FCI must lie below RHF"


def test_analytic_energies_reproduce_the_record(record, analytic):
    nuc = 1.0 / (RECORD_D * BOHR_PER_ANGSTROM)
    e_hf = hf_from(analytic["h1"], analytic["h2"], nuc)
    e_fci = fci_from(analytic["h1"], analytic["h2"], nuc)
    assert abs(e_hf - float(record["hf_energy"])) < 1e-10
    assert abs(e_fci - float(record["fci_energy"])) < 1e-10


def test_the_quoted_literature_pair_is_not_one_geometry():
    """CH-12's two published numbers cannot both be H2/STO-3G.

    ``-1.11765865`` (HF) lies below the STO-3G HF minimum at *any* bond
    length, so it is not this basis.  ``-1.13727017`` (FCI) is reproduced at
    R = 0.74140 A -- the experimental equilibrium distance -- i.e. at a
    different geometry than the record's 0.735 A.  Neither can referee the
    record, which is why the record is held to the recomputation above.
    """
    lit_hf, lit_fci = -1.11765865, -1.13727017
    r_hf_min, e_hf_min = golden_min(lambda r: energies(r)[0], 0.68, 0.76)
    assert lit_hf < e_hf_min, (
        f"the STO-3G HF minimum is {e_hf_min:.10f} at R = {r_hf_min:.5f} A, so "
        f"{lit_hf} is below every bond length and cannot be this basis")

    assert energies(RECORD_D)[1] - lit_fci < 0.0 < energies(0.80)[1] - lit_fci, (
        "the literature FCI must lie between the values at 0.735 and 0.80 A")
    r_upper = bisect(lambda r: energies(r)[1] - lit_fci, RECORD_D, 0.80)
    assert abs(r_upper - 0.7414) < 0.002, (
        f"the literature FCI is reproduced at R = {r_upper:.5f} A; if that is "
        f"the equilibrium distance the pair is geometry-mismatched, not wrong")
    e_hf_there = energies(r_upper)[0]
    assert e_hf_there - lit_hf > 5e-4, (
        f"at R = {r_upper:.5f} A the STO-3G HF energy is {e_hf_there:.10f}, "
        f"which is {e_hf_there - lit_hf:+.2e} from {lit_hf}: one geometry "
        f"cannot produce both published numbers")
