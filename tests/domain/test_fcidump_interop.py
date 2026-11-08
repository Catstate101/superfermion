"""FCIDUMP interop with the two readers that disagree with the standard.

Measured with the published 0.1.27 wheel: PySCF 2.14.0 (``tools.fcidump.to_scf``)
and Qiskit Nature 0.7.2 (``fcidump_to_problem``) both parse ``NOTATION=`` as an
unknown namelist key, drop it, and take the four indices as chemist ``(ij|kl)``
-- the layout their own writers emit.  A default, physicist-labelled file from
those releases was therefore read transposed and silent: -1.967656 Ha instead
of -1.137306 Ha on H2/STO-3G, and -124.099 Ha instead of -75.013 Ha on
water/STO-3G, with no error on either side.

0.1.28 labels chemist by default for that reason.  These tests pin the decision
in both directions: our default output must come back from PySCF untransposed,
and a physicist file must still be readable by us while being visibly wrong
there.  The PySCF half skips when PySCF is absent; the ``chemistry-platforms``
lane installs it.

The reverse direction is closed in 0.1.28.  Qiskit Nature rebuilds the
``&FCI`` header from its own dataclass when it writes, so the ``NOTATION=``
variable does not survive a file that has been through ``FCIDump.to_file``.
Its header is also padded differently from PySCF's (``MS2=   0`` against
``MS2=0``, ``NELEC=   2`` against ``NELEC= 2``, no leading space), which kept
such a file out of the legacy fingerprint.  Those bytes come from
``FCIDump.to_file``'s source and were then measured on qiskit-nature 0.7.2, so
they are now a fingerprint of their own; ``test_a_qiskit_rewrite_is_still_read``
pins the round trip.  Files that match no writer at all fall back to chemist
rather than physicist, because that is the convention every documented writer
uses -- NECI's manual on the Molpro/Molcas transfer, HANDE's format notes,
HORTON, orbkit and Psi4 included.

The header also carries ``ORBSYM`` and ``ISYM``.  Those are labels rather than
integrals, but 0.1.28 keeps them: the reader now records them on
``MolecularData.fcidump_orbsym``/``fcidump_isym``, and both referees were
measured returning the values we wrote (PySCF via ``tools.fcidump.read``,
Qiskit Nature via ``FCIDump.from_file``).  The writer also no longer accepts a
tensor that the eight-fold symmetry cannot describe without saying so: one
record per orbit is emitted, so anything outside the canonical quarter is
dropped, and ``validate=True`` turns that from a silent loss into a warning.
"""

import warnings

import numpy as np
import pytest

from superfermion.chemistry import fcidump
from superfermion.chemistry.pyscf_bridge import MolecularData

pytestmark = pytest.mark.domain

try:  # PySCF is an optional extra; only the interop half below needs it.
    from pyscf import ao2mo, fci, gto, scf
    from pyscf.tools import fcidump as pfcidump

    _HAVE_PYSCF = True
except Exception:  # pragma: no cover - exercised on lanes without PySCF
    _HAVE_PYSCF = False

try:  # Qiskit Nature is the second referee; not a project dependency.
    import scipy.sparse.linalg as _spla
    from qiskit_nature.second_q.formats.fcidump import FCIDump
    from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    _HAVE_QISKIT_NATURE = True
except Exception:  # pragma: no cover - exercised on lanes without Qiskit
    _HAVE_QISKIT_NATURE = False

needs_pyscf = pytest.mark.skipif(not _HAVE_PYSCF, reason="pyscf is not installed")
needs_both = pytest.mark.skipif(
    not (_HAVE_PYSCF and _HAVE_QISKIT_NATURE),
    reason="pyscf and qiskit-nature are both needed",
)


_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _synthetic(n_orbs: int = 3, seed: int = 41):
    """Synthetic integrals with all eight chemist symmetries."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_orbs, n_orbs))
    h1 = a + a.T
    raw = rng.normal(size=(n_orbs,) * 4) / 4.0
    h2 = sum(raw.transpose(p) for p in _PERMUTATIONS) / 8.0
    return h1, h2


def _h2_sto3g():
    """H2/STO-3G: integrals, and PySCF's own FCI energy as the reference."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    coeff = mf.mo_coeff
    norb = coeff.shape[1]
    h1 = coeff.T @ mf.get_hcore() @ coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, coeff), norb)
    return mol, mf, h1, eri, fci.FCI(mf).kernel()[0]


def _score_in_pyscf(path):
    """Read ``path`` the way a PySCF user does, and solve FCI on the result."""
    mf = pfcidump.to_scf(str(path))
    # ``to_scf`` returns a bare Mole with ``nao`` set but no basis, so
    # ``nao_nr()`` reports 0; the attribute is the orbital count here.
    norb = mf.mol.nao
    eri = ao2mo.restore(1, mf._eri, norb)
    nelec = mf.mol.nelectron
    energy = fci.direct_spin1.kernel(
        mf.get_hcore(), eri, norb, (nelec // 2, nelec // 2)
    )[0]
    return energy + mf.energy_nuc(), eri


def _records(text):
    """The (value, i, j, k, l) records of a file, parsed without ``fcidump``.

    Written from the format description rather than by calling the module under
    test, so a writer/reader pair that agrees with itself but not with the
    standard is still caught.
    """
    body = text.split("&END", 1)[1]
    out = []
    for line in body.strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append((float(parts[0]), *(int(p) for p in parts[1:])))
    return out


def _decode_two_electron(text, norb):
    """Rebuild ``(pq|rs)`` from the records via the standard rule.

    The FCIDUMP standard stores ``<ij|kl>`` under the labels ``i j k l``, and
    ``(pq|rs) == <pr|qs>``, so the value labelled ``(i, j, k, l)`` belongs at
    ``(i, k, j, l)`` in the chemist tensor.  Only the canonical quarter of the
    tensor is written -- the file relies on the integral's eight symmetries --
    so each record fills its permutation family, as a standard reader must.
    """
    h2 = np.zeros((norb,) * 4)
    for value, i, j, k, l in _records(text):
        if not (k and l):
            continue
        for p, q, r, s in _PERMUTATIONS:
            idx = (i - 1, k - 1, j - 1, l - 1)
            h2[idx[p], idx[q], idx[r], idx[s]] = value
    return h2


def _decode_chemist(text, norb):
    """Rebuild ``(pq|rs)`` treating the labels as the chemist indices."""
    h2 = np.zeros((norb,) * 4)
    for value, i, j, k, l in _records(text):
        if not (k and l):
            continue
        for p, q, r, s in _PERMUTATIONS:
            idx = (i - 1, j - 1, k - 1, l - 1)
            h2[idx[p], idx[q], idx[r], idx[s]] = value
    return h2


class TestTheWriterDefault:
    """0.1.28 writes chemist labels, matching PySCF and Qiskit Nature."""

    def test_the_default_output_is_labelled_chemist(self):
        h1, h2 = _synthetic()
        text = fcidump.write_fcidump_text((h1, h2, 0.3), n_elec=2)
        assert "NOTATION=chemist," in text
        assert fcidump.detect_fcidump_notation(text)[0] == "chemist"

    def test_the_default_round_trip_is_exact(self):
        h1, h2 = _synthetic()
        back = fcidump.read_fcidump_text(
            fcidump.write_fcidump_text((h1, h2, 0.3), n_elec=2)
        )
        assert back.fcidump_notation == "chemist"
        assert np.allclose(back.h1, h1, atol=1e-12)
        assert np.allclose(back.h2, h2, atol=1e-12)

    def test_an_unannotated_default_is_still_read_as_chemist(self):
        """``annotate=False`` must not turn our own default into a legacy file."""
        h1, h2 = _synthetic()
        text = fcidump.write_fcidump_text(
            (h1, h2, 0.3), n_elec=2, annotate=False
        )
        assert "NOTATION" not in text
        assert fcidump.detect_fcidump_notation(text)[0] == "chemist"
        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump_text(text)
        assert np.allclose(back.h2, h2, atol=1e-12)

    def test_an_unannotated_physicist_file_stays_physicist(self):
        """The one-space header keeps physicist output out of the fingerprint."""
        h1, h2 = _synthetic()
        text = fcidump.write_fcidump_text(
            (h1, h2, 0.3), n_elec=2, notation="physicist", annotate=False
        )
        assert fcidump.detect_fcidump_notation(text)[0] == "physicist"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = fcidump.read_fcidump_text(text)
        assert [str(w.message) for w in caught] == []
        assert back.fcidump_notation == "physicist"
        assert np.allclose(back.h2, h2, atol=1e-12)


class TestTheStandardRule:
    """The standard layout, decoded from the format description, not the reader.

    Molpro itself cannot be installed here, so the physicist path is checked
    against the rule it implements: a record labelled ``(i, j, k, l)`` holds
    ``<ij|kl>``, and ``<ij|kl> == (ik|jl)``.  Decoding the text with that rule
    alone catches a writer and reader that agree with each other but not with
    the standard -- a pairing every round-trip test above would pass.
    """

    def test_the_labels_swap_the_middle_indices_for_physicist(self):
        """The pair that changes place between layouts, named by its records.

        This tensor has two two-electron records, ``(11|00) = 0.25`` and
        ``(10|10) = 0.75`` (0-based), and the writer emits them under fixed
        labels: the standard formula ``lab = (i+1, k+1, j+1, l+1)`` puts 0.25
        at ``2 1 2 1`` and 0.75 at ``2 2 1 1``, while the chemist formula
        ``lab = (i+1, j+1, k+1, l+1)`` puts them the other way round.  Reading
        with the wrong convention therefore sees 0.25 where 0.75 belongs, in
        every entry the two share.
        """
        h1 = np.zeros((2, 2))
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 0, 1, 1] = h2[1, 1, 0, 0] = 0.25
        for i, j, k, l in ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)):
            h2[i, j, k, l] = 0.75

        def labels(text):
            return {
                (i, j, k, l): value
                for value, i, j, k, l in _records(text)
                if k and l
            }

        physicist = labels(
            fcidump.write_fcidump_text((h1, h2, 0.0), n_elec=2, notation="physicist")
        )
        assert physicist[(2, 2, 1, 1)] == pytest.approx(0.75)
        assert physicist[(2, 1, 2, 1)] == pytest.approx(0.25)

        chemist = labels(fcidump.write_fcidump_text((h1, h2, 0.0), n_elec=2))
        assert chemist[(2, 2, 1, 1)] == pytest.approx(0.25)
        assert chemist[(2, 1, 2, 1)] == pytest.approx(0.75)

        # Same two numbers either way: only the label travelling with each one
        # is different, which is why a transposed read is not detectable from
        # the record values alone.
        assert sorted(physicist.values()) == sorted(chemist.values())

    def test_a_physicist_file_decodes_with_the_standard_rule(self):
        h1, h2 = _synthetic(4, seed=7)
        text = fcidump.write_fcidump_text((h1, h2, 0.1), n_elec=2, notation="physicist")
        assert np.allclose(_decode_two_electron(text, 4), h2, atol=1e-12)

    def test_a_chemist_file_decodes_as_itself(self):
        h1, h2 = _synthetic(4, seed=8)
        text = fcidump.write_fcidump_text((h1, h2, 0.1), n_elec=2)
        assert np.allclose(_decode_chemist(text, 4), h2, atol=1e-12)


_HORTON_TEXT = (
    "&FCI NORB=2,NELEC=2,MS2=0,\n"
    "ORBSYM= 1,1,\n"
    "ISYM=1\n"
    "&END\n"
    " 7.500000000000000000e-01    2    1    2    1\n"
    " 2.500000000000000000e-01    2    2    1    1\n"
    " 2.000000000000000000e+00    0    0    0    0\n"
)


class TestADocumentedThirdWriter:
    """HORTON's published FCIDUMP header, run through this detector.

    No HORTON installation is available here, so the bytes are transcribed from
    HORTON 2.0.1's documentation rather than produced by it: ``&FCI`` at column
    0 with unpadded values, ``ORBSYM= 1,1,...`` with a space after the ``=``,
    ``ISYM=1`` with no trailing comma, ``&END`` on its own line, and records in
    ``%.16e``.  The orbital count is cut to two so the two labellings disagree
    on the same record, which the 28-orbital example cannot show cheaply.

    HORTON's own manual states that its indices are chemist ``(ij|kl)``, so the
    chemist reading below is what its writer means -- and it shares PySCF's
    header shape, so it also inherits the pre-0.1.26 warning.  That is the one
    third-party sample available here.  Molpro's writer could not be run, but
    NECI's manual states that the FCIDUMP Molpro produces uses the chemist
    notation, so it lands on the same reading through the chemist default.
    """

    def test_the_published_header_is_recognised(self):
        notation, reason = fcidump.detect_fcidump_notation(_HORTON_TEXT)
        assert notation == "chemist"
        assert reason.startswith("PySCF writer fingerprint")

    def test_the_unusual_header_variants_still_parse(self):
        """No comma after ``ISYM``, a space after ``ORBSYM=``, no indent."""
        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump_text(_HORTON_TEXT)

        assert back.n_orbitals == 2
        assert back.n_electrons == 2
        assert back.nuclear_repulsion == pytest.approx(2.0)
        assert back.fcidump_orbsym == (1, 1)
        assert back.fcidump_isym == 1
        assert back.fcidump_notation == "chemist"

    def test_the_variants_are_parsed_and_not_defaulted(self):
        """``ISYM=1`` and all-ones ``ORBSYM`` are also the defaults.

        The sample above therefore cannot tell a parsed value from a defaulted
        one, so the same header shape is asked for labels that could only have
        come out of it.
        """
        text = _HORTON_TEXT.replace("ORBSYM= 1,1,", "ORBSYM= 2,1,").replace(
            "ISYM=1\n", "ISYM=3\n"
        )
        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump_text(text)

        assert back.fcidump_orbsym == (2, 1)
        assert back.fcidump_isym == 3

    def test_a_wrong_length_orbsym_is_not_carried_forward(self):
        """A malformed count is dropped, not re-emitted or raised on later."""
        text = _HORTON_TEXT.replace("ORBSYM= 1,1,", "ORBSYM= 1,")
        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump_text(text)

        assert back.fcidump_orbsym is None
        assert "  ORBSYM=1,1," in fcidump.write_fcidump_text(back)

    def test_the_chemist_reading_is_the_one_horton_documents(self):
        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump_text(_HORTON_TEXT)

        # Records ``2 1 2 1`` and ``2 2 1 1``: chemist puts 0.75 at (i0|i0)
        # and 0.25 at (ii|00); the standard reading swaps them.
        assert back.h2[1, 0, 1, 0] == pytest.approx(0.75)
        assert back.h2[1, 1, 0, 0] == pytest.approx(0.25)
        assert back.h2[0, 1, 0, 1] == pytest.approx(0.75)


class TestTheNotationDecision:
    """What ``notation="auto"`` decides when a file carries no marker."""

    def test_an_unannotated_physicist_file_is_recognised_by_its_space(self):
        """0.1.26/0.1.27 wrote ``MS2=0 ,`` for physicist output.

        That one space is the whole difference between those files and every
        other writer, and it now matters: an unmarked file defaults to chemist,
        so without this rule the physicist output of the two releases
        immediately before this one would be read transposed.
        """
        h1, h2 = _synthetic(3, seed=31)
        text = fcidump.write_fcidump_text(
            (h1, h2, 0.0), n_elec=2, notation="physicist", annotate=False
        )
        assert "MS2=0 ," in text

        notation, reason = fcidump.detect_fcidump_notation(text)
        assert notation == "physicist"
        assert reason.startswith("Superfermion 0.1.26-0.1.27")

    def test_a_psi4_shaped_header_falls_to_chemist(self):
        """Psi4 documents spaces after the commas, which no fingerprint claims.

        The header below is transcribed from Psi4's documentation.  It matched
        no fingerprint before 0.1.28 either, and used to land on the physicist
        labelling -- transposing a writer whose convention is chemist.
        """
        text = (
            "&FCI NORB=2, NELEC=2, MS2=0, UHF=.FALSE., ORBSYM=1,5, ISYM=1,\n"
            "&END\n"
            " 5.0000000000000000e-01    1    1    1    1\n"
        )
        notation, reason = fcidump.detect_fcidump_notation(text)
        assert notation == "chemist"
        assert "no writer marker" in reason

    def test_the_pre_0126_header_cannot_be_told_from_pyscfs(self):
        """Written from the v0.1.25 tag, pinned as a documented impossibility.

        Superfermion <= 0.1.25 used the same ``%4d``/``%2d``/``%d`` widths and
        the same ``ORBSYM``/``ISYM``/``&END`` lines PySCF's ``write_head`` uses,
        so its physicist files and PySCF's chemist ones differ in the integrals
        only.  No rule can separate them; the warning is the resolution, and
        this test exists so the equality is not mistaken for a missing rule.
        """
        norb, nelec, spin = 2, 2, 0
        legacy = (
            " &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n" % (norb, nelec, spin)
            + "  ORBSYM=" + ",".join(["1"] * norb) + ",\n"
            + "  ISYM=1,\n"
            + " &END\n"
        )
        assert legacy == (
            " &FCI NORB=   2,NELEC= 2,MS2=0,\n"
            "  ORBSYM=1,1,\n"
            "  ISYM=1,\n"
            " &END\n"
        )


class TestTheSymmetryFields:
    """``ORBSYM``/``ISYM`` are labels, and a round trip must not drop them."""

    def test_orbsym_and_isym_survive_a_round_trip(self):
        h1, h2 = _synthetic(4, seed=11)
        text = fcidump.write_fcidump_text(
            (h1, h2, 0.3), n_elec=4, orbsym=[1, 2, 3, 4], isym=2
        )
        assert "  ORBSYM=1,2,3,4," in text
        assert "  ISYM=2," in text

        back = fcidump.read_fcidump_text(text)
        assert back.fcidump_orbsym == (1, 2, 3, 4)
        assert back.fcidump_isym == 2

    def test_a_write_back_keeps_the_header_it_read(self):
        """Until 0.1.28 the labels were parsed and then dropped.

        ``read_fcidump`` kept them in a local, ``MolecularData`` had no field
        for them and the writer defaulted to all-ones, so copying a file lost
        the symmetry labels that a sector calculation needs.
        """
        h1, h2 = _synthetic(4, seed=12)
        original = fcidump.write_fcidump_text(
            (h1, h2, 0.3), n_elec=4, orbsym=[1, 2, 3, 4], isym=2
        )
        copy = fcidump.write_fcidump_text(fcidump.read_fcidump_text(original))
        assert "  ORBSYM=1,2,3,4," in copy
        assert "  ISYM=2," in copy

    def test_a_numpy_orbsym_is_accepted(self):
        """``orbsym or [1] * norb`` raised on an array's ambiguous truth."""
        h1, h2 = _synthetic(3, seed=13)
        text = fcidump.write_fcidump_text(
            (h1, h2, 0.0), n_elec=2, orbsym=np.array([2, 1, 3])
        )
        assert "  ORBSYM=2,1,3," in text

    def test_an_orbsym_outside_the_format_range_is_reported(self):
        """0 is legal -- HANDE uses it for "no well defined symmetry".

        Anything past 8 is a writer bug or 0-based irrep ids, and it used to be
        written and read back without a word.
        """
        h1, h2 = _synthetic(3, seed=32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcidump.write_fcidump_text(
                (h1, h2, 0.0), n_elec=2, orbsym=[0, 1, 8]
            )

        with pytest.warns(UserWarning, match="outside the 0..8 range"):
            text = fcidump.write_fcidump_text(
                (h1, h2, 0.0), n_elec=2, orbsym=[0, 1, 9]
            )
        assert "  ORBSYM=0,1,9," in text

        with pytest.warns(UserWarning, match="outside the 0..8 range"):
            back = fcidump.read_fcidump_text(text)
        assert back.fcidump_orbsym == (0, 1, 9)

    def test_orbsym_of_the_wrong_length_is_rejected(self):
        h1, h2 = _synthetic(3, seed=14)
        with pytest.raises(ValueError, match="3 orbitals"):
            fcidump.write_fcidump_text((h1, h2, 0.0), n_elec=2, orbsym=[1, 1])

    def test_h2_of_the_wrong_shape_is_rejected(self):
        h1, h2 = _synthetic(3, seed=15)
        with pytest.raises(ValueError, match=r"\(3, 3, 3, 3\)"):
            fcidump.write_fcidump_text((h1, h2[:2], 0.0), n_elec=2)

    def test_a_symmetric_tensor_warns_about_nothing(self):
        h1, h2 = _synthetic(3, seed=16)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcidump.write_fcidump_text((h1, h2, 0.0), n_elec=2)

    def test_an_asymmetric_tensor_is_named_rather_than_dropped(self):
        """Only the canonical quarter is written, so the rest must be flagged.

        Measured here: lifting the non-canonical image ``(0,1,0,0)`` by 0.5
        leaves the file describing the unamended value there, which used to
        happen with no signal at all.
        """
        h1, h2 = _synthetic(4, seed=17)
        broken = h2.copy()
        broken[0, 1, 0, 0] += 0.5

        with pytest.warns(UserWarning, match="eight-fold"):
            text = fcidump.write_fcidump_text((h1, broken, 0.0), n_elec=4)

        back = fcidump.read_fcidump_text(text)
        assert back.h2[0, 1, 0, 0] == pytest.approx(h2[0, 1, 0, 0])
        assert back.h2[1, 0, 0, 0] == pytest.approx(h2[1, 0, 0, 0])

    def test_the_check_can_be_switched_off(self):
        h1, h2 = _synthetic(3, seed=18)
        broken = h2.copy()
        broken[1, 0, 0, 0] += 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcidump.write_fcidump_text(
                (h1, broken, 0.0), n_elec=2, validate=False
            )


@needs_pyscf
class TestPySCF:
    """PySCF is the referee: it must agree with our default and not with the
    physicist labelling, which is the deviation this default exists to dodge."""

    def test_the_default_file_reproduces_pyscfs_own_fci(self, tmp_path):
        mol, mf, h1, eri, e_ref = _h2_sto3g()
        path = tmp_path / "h2_default.fcidump"
        fcidump.write_fcidump(
            (h1, eri, mf.energy_nuc()), str(path),
            n_elec=mol.nelectron, spin=0,
        )

        energy, eri_back = _score_in_pyscf(path)

        assert np.allclose(eri_back, eri, atol=1e-9)
        assert energy == pytest.approx(e_ref, abs=1e-9)

    def test_a_physicist_file_is_transposed_by_pyscf(self, tmp_path):
        mol, mf, h1, eri, e_ref = _h2_sto3g()
        path = tmp_path / "h2_physicist.fcidump"
        fcidump.write_fcidump(
            (h1, eri, mf.energy_nuc()), str(path),
            n_elec=mol.nelectron, spin=0, notation="physicist",
        )

        energy, eri_back = _score_in_pyscf(path)

        # Not an approximation error: J and K have been exchanged.  Measured
        # on this geometry the FCI energy moves by 0.358 Ha.
        assert not np.allclose(eri_back, eri, atol=1e-6)
        assert abs(energy - e_ref) > 0.1

    def test_a_file_written_by_pyscf_reads_back_chemist(self, tmp_path):
        mol, mf, _h1, eri, _e_ref = _h2_sto3g()
        path = tmp_path / "h2_pyscf.fcidump"
        pfcidump.from_mo(mol, str(path), mf.mo_coeff)

        with pytest.warns(UserWarning, match="0.1.25"):
            back = fcidump.read_fcidump(path)

        assert back.fcidump_notation == "chemist"
        assert np.allclose(back.h2, eri, atol=1e-9)

    def test_orbsym_reaches_pyscfs_reader(self, tmp_path):
        """Measured: PySCF 2.14.0 parses the values, not merely the line."""
        h1, h2 = _synthetic(4, seed=21)
        path = tmp_path / "sym.fcidump"
        fcidump.write_fcidump(
            (h1, h2, 0.0), str(path), n_elec=4, orbsym=[1, 2, 3, 4], isym=2
        )

        parsed = pfcidump.read(str(path))

        assert list(parsed["ORBSYM"]) == [1, 2, 3, 4]
        assert int(parsed["ISYM"]) == 2

    def test_pyscfs_header_is_byte_identical_to_the_legacy_writers(self, tmp_path):
        """The measured reason a pre-0.1.26 file cannot be detected."""
        mol, mf, _h1, _eri, _e_ref = _h2_sto3g()
        path = tmp_path / "h2_pyscf.fcidump"
        pfcidump.from_mo(mol, str(path), mf.mo_coeff)

        head = path.read_text(encoding="utf-8").split("&END", 1)[0] + "&END\n"

        assert head == (
            " &FCI NORB=   2,NELEC= 2,MS2=0,\n"
            "  ORBSYM=1,1,\n"
            "  ISYM=1,\n"
            " &END\n"
        )

    def test_a_real_eri_tensor_writes_without_a_symmetry_warning(self, tmp_path):
        """``ao2mo.restore`` is not bitwise symmetric under the pair swap.

        Measured on this geometry: equality holds bitwise under ``(p<->q)`` and
        ``(r<->s)``, and the pair swap differs by 1.7e-16.  That is why the
        check is relative to the largest element instead of a fixed 1e-12 --
        the first assertion here would start failing if PySCF ever returned an
        exactly symmetric tensor, and the second is the contract that matters.
        """
        _mol, _mf, h1, eri, _e_ref = _h2_sto3g()
        assert not np.array_equal(eri, eri.transpose(2, 3, 0, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcidump.write_fcidump(
                (h1, eri, 0.0), str(tmp_path / "eri.fcidump"), n_elec=2
            )


@needs_both
class TestQiskitNature:
    """Qiskit Nature 0.7.x, the second referee: H2/STO-3G through JW + ARPACK."""

    @staticmethod
    def _energy(path):
        problem = fcidump_to_problem(FCIDump.from_file(str(path)))
        qubit_op = JordanWignerMapper().map(problem.second_q_ops()[0])
        matrix = qubit_op.to_matrix(sparse=True)
        lowest = _spla.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
        return lowest + sum(problem.hamiltonian.constants.values()), qubit_op.num_qubits

    def _write(self, path, notation=None):
        mol, mf, h1, eri, e_ref = _h2_sto3g()
        kwargs = {} if notation is None else {"notation": notation}
        fcidump.write_fcidump(
            (h1, eri, mf.energy_nuc()), str(path),
            n_elec=mol.nelectron, spin=0, **kwargs,
        )
        return eri, e_ref

    def test_the_default_file_gives_the_reference_energy(self, tmp_path):
        path = tmp_path / "h2_default.fcidump"
        _eri, e_ref = self._write(path)

        energy, n_qubits = self._energy(path)

        assert n_qubits == 4
        assert energy == pytest.approx(e_ref, abs=1e-6)

    def test_a_physicist_file_is_transposed_there(self, tmp_path):
        path = tmp_path / "h2_physicist.fcidump"
        _eri, e_ref = self._write(path, notation="physicist")

        energy, _n_qubits = self._energy(path)

        # Measured on this geometry: -1.971826 Ha against -1.137284 Ha.
        assert abs(energy - e_ref) > 0.5

    def test_a_qiskit_rewrite_is_still_read(self, tmp_path):
        """The marker does not survive ``to_file``; the padding does.

        Qiskit Nature writes ``&FCI NORB=   2,NELEC=   2,MS2=   0,`` -- no
        leading space, every integer padded to four columns -- and drops the
        ``NOTATION=`` line.  Before 0.1.28 that rewrite fell through to the
        physicist default and came back transposed; the padding is now a
        fingerprint in its own right, so the automatic read is correct.
        """
        source = tmp_path / "h2_default.fcidump"
        eri, _e_ref = self._write(source)
        rewritten = tmp_path / "h2_qiskit.fcidump"
        FCIDump.from_file(str(source)).to_file(str(rewritten))

        text = rewritten.read_text(encoding="utf-8")
        assert "NOTATION" not in text
        assert text.startswith("&FCI NORB="), "qiskit does not indent the header"
        assert "NELEC=   2" in text, "qiskit pads NELEC to four columns"

        assert fcidump.detect_fcidump_notation(text)[0] == "chemist"
        auto = fcidump.read_fcidump(rewritten)
        assert auto.fcidump_notation == "chemist"
        assert np.allclose(auto.h2, eri, atol=1e-9)

        # The data really is unchanged as well, so this is a labelling fix and
        # not a coincidence: decoding qiskit's own records with the chemist
        # rule gives back the tensor we started from.
        assert np.allclose(_decode_chemist(text, eri.shape[0]), eri, atol=1e-9)

    def test_orbsym_reaches_qiskits_reader(self, tmp_path):
        """Measured: Qiskit Nature 0.7.2 parses both values from our header."""
        h1, h2 = _synthetic(4, seed=22)
        path = tmp_path / "sym.fcidump"
        fcidump.write_fcidump(
            (h1, h2, 0.0), str(path), n_elec=4, orbsym=[1, 2, 3, 4], isym=2
        )

        dump = FCIDump.from_file(path)

        assert dump.orbsym == [1, 2, 3, 4]
        assert dump.isym == 2


class TestMolecularDataIsUnaffected:
    """The in-memory container already stored chemist; only the file changed."""

    def test_h2_stays_chemist_whatever_the_file_held(self):
        h1, h2 = _synthetic()
        mol = MolecularData(
            name="model", geometry="H 0 0 0", h1=h1, h2=h2,
            nuclear_repulsion=0.3, n_orbitals=3, n_electrons=2,
        )
        physicist = fcidump.write_fcidump_text(mol, notation="physicist")
        back = fcidump.read_fcidump_text(physicist)
        assert back.fcidump_notation == "physicist"
        assert np.allclose(back.h2, mol.h2, atol=1e-12)
