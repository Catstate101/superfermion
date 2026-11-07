"""Regressions for the seven defects fixed after v0.1.24.

Every test here fails against the 0.1.24 code and passes after the fix, so the
file doubles as the record of what was wrong:

1. **Seed mixing** (Rust, ``crate::mix_seed``).  The per-shot and per-block
   samplers derived their RNG from ``seed.wrapping_add(index)``, so seeds ``s``
   and ``s+1`` shared all but one of the 63 blocks a 64-shot block sampler
   draws.  Two "independent" runs with adjacent seeds were therefore almost the
   same run, and a spread measured over ``range(n)`` collapsed to a fraction of
   the binomial prediction.  Pinned here as histogram correlation between
   adjacent seeds, which is what a user samples.
2. **FCIDUMP notation.**  ``read_fcidump`` always assumed the FCIDUMP standard
   (physicist) ordering, so a file written by PySCF -- whose writer emits
   chemist-ordered values under the same labels -- was read transposed.  On the
   LiH file used during the fix that turned -7.8823 Ha into -10.4717 Ha.
3. **BP+OSD ``osd_order``.**  ``osd_order`` was inert: OSD-1 flipped *pivot*
   bits and then re-tested the syndrome, which can never match, so every value
   of the parameter returned the same error.  It now enumerates free-variable
   patterns, which is a real budget: ``osd_order >= n_free`` is exact
   minimum-weight decoding.
4. **AVAS selection basis.**  The projector was built against the working
   basis, so adding polarization functions changed the answer (water/STO-3G
   with the O target kept 6 orbitals where PySCF keeps 5, and the gap grows
   with the basis).  PySCF projects the ``minao`` functions instead.
5. **XYZ parsing.**  The reader took ``lines[1]`` as the title and everything
   after ``lines[2]`` as atoms, so a wrong count was never noticed and an
   unparsable atom line surfaced as a PySCF error with no line number.
6. **SCF convergence.**  ``MolecularData`` recorded no convergence flag, so a
   run that stopped at iteration 20 was indistinguishable from a converged one.
7. **AEGISS.**  The entropy refinement raised ``ValueError`` when every orbital
   was below the threshold, which is the *default* outcome for water with the O
   target.  It now returns the unrefined AVAS space with
   ``refinement_applied=False`` and a ``UserWarning``.
8. **MPS ``bond_dim``.**  ``bond_dim=0`` was accepted and silently produced the
   uniform distribution: the QR cap becomes ``min(0, rank) = 0``, every bond
   collapses to a single state.  Measured on GHZ-3, all eight outcomes came back
   at ~25 counts each out of 200.  Values below 1 are now refused, and the
   refusal names the argument.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion.chemistry.fcidump import (
    detect_fcidump_notation,
    read_fcidump_text,
    write_fcidump_text,
)
from superfermion.chemistry.pyscf_bridge import (
    MolecularData,
    molecule_from_geometry,
    molecule_from_xyz,
    parse_xyz_text,
)
from superfermion.qec.decoders import BPOSD_Decoder

pytestmark = pytest.mark.domain


# ═══════════════════════════════════════════════════════════════════════════
# 1. Seed mixing
# ═══════════════════════════════════════════════════════════════════════════

def _uniform(n_qubits: int = 4) -> "sf.Circuit":
    circuit = sf.Circuit(n_qubits)
    for qubit in range(n_qubits):
        circuit.h(qubit)
        circuit.measure(qubit)
    return circuit


def _histogram(result, n_qubits: int = 4) -> np.ndarray:
    return np.array(
        [result.counts.get(format(i, f"0{n_qubits}b"), 0) for i in range(2**n_qubits)],
        dtype=float,
    )


class TestSeedsDoNotShareRandomness:
    """Adjacent seeds must not be near-copies of each other."""

    @staticmethod
    def _adjacent_correlations(method: str, n_seeds: int = 16) -> np.ndarray:
        circuit = _uniform()
        hists = [
            _histogram(sf.run(circuit, method=method, shots=1024, seed=seed))
            for seed in range(n_seeds)
        ]
        correlations = []
        for left, right in zip(hists, hists[1:]):
            # 16 outcomes with ~64 counts each: independent runs give a
            # correlation with sd ~0.26, a shared RNG gives ~0.94.
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
        return np.array(correlations)

    @pytest.mark.parametrize("method", ["stabilizer", "mps"])
    @pytest.mark.timeout(300)
    def test_adjacent_seeds_produce_independent_histograms(self, method):
        correlations = self._adjacent_correlations(method)
        assert correlations.mean() < 0.5, (
            f"{method}: mean adjacent-seed correlation {correlations.mean():.3f}; "
            "the seeds are sharing random streams"
        )

    @pytest.mark.parametrize("method", ["stabilizer", "mps"])
    def test_the_same_seed_still_reproduces(self, method):
        """Mixing must not cost determinism."""
        circuit = _uniform()
        first = sf.run(circuit, method=method, shots=256, seed=13)
        second = sf.run(circuit, method=method, shots=256, seed=13)
        assert first.counts == second.counts


# ═══════════════════════════════════════════════════════════════════════════
# 2. FCIDUMP notation
# ═══════════════════════════════════════════════════════════════════════════

# PySCF's `tools.fcidump.write_head` shape: the comma is butted against the
# value, which is the fingerprint `detect_fcidump_notation` keys on.  Each
# integral line is `value i j k l` -- value first, as the format requires.
_PYSCF_TEXT = (
    " &FCI NORB=   2,NELEC= 2,MS2=0,\n"
    "  ORBSYM=1,1,\n"
    "  ISYM=1,\n"
    " &END\n"
    "1.0 1 1 1 1\n"     # <11|11> == (11|11), notation-independent
    "0.5 2 2 1 1\n"     # chemist: (22|11);  physicist: <22|11> = (21|21)
    "-1.25 1 1  0  0\n"
    "0.7 0 0 0 0\n"
)

# A value small enough for %.16g to print in exponent form, which is what
# PySCF really does: measured 128 such lines out of 547 on water/STO-3G.
_SMALL_VALUE = 4.040691484975394e-05
_SMALL_VALUE_TEXT = _PYSCF_TEXT.replace("0.5 2 2 1 1", f"{_SMALL_VALUE!r} 2 2 1 1")

# The eight permutations a chemist-notation ERI is invariant under.  Building
# the test tensor from these (rather than setting entries by hand) matters: the
# reader reconstructs all eight from each canonical entry, so a tensor that
# breaks the symmetry would fail the round trip for reasons that have nothing
# to do with the notation.
_ERI_PERMS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _symmetrise(tensor: np.ndarray) -> np.ndarray:
    return sum(tensor.transpose(p) for p in _ERI_PERMS) / 8.0


class TestFcidumpNotationDetection:
    """The reader must not assume one writer's convention."""

    def test_a_pyscf_shaped_file_is_read_as_chemist(self):
        notation, reason = detect_fcidump_notation(_PYSCF_TEXT)
        assert notation == "chemist"
        assert "PySCF" in reason

        # The header is also what Superfermion <= 0.1.25 wrote, so the read has
        # to say so instead of choosing silently.
        with pytest.warns(UserWarning, match="0.1.25"):
            mol = read_fcidump_text(_PYSCF_TEXT)
        assert mol.fcidump_notation == "chemist"
        assert mol.h2[1, 1, 0, 0] == pytest.approx(0.5)
        assert mol.h2[1, 0, 1, 0] == pytest.approx(0.0)

    def test_small_values_in_exponent_form_do_not_break_detection(self):
        """%.16g prints ``e-05``; that is PySCF output, not a foreign writer.

        An earlier version of the detector treated an exponent field as
        evidence of a Fortran writer and flipped such files back to
        ``"physicist"``, which is the transposition the detection exists to
        prevent.
        """
        notation, reason = detect_fcidump_notation(_SMALL_VALUE_TEXT)
        assert notation == "chemist"
        assert "PySCF" in reason

        with pytest.warns(UserWarning, match="0.1.25"):
            mol = read_fcidump_text(_SMALL_VALUE_TEXT)
        assert mol.h2[1, 1, 0, 0] == pytest.approx(_SMALL_VALUE)

    def test_a_file_with_no_writer_evidence_defaults_to_the_standard(self):
        plain = _PYSCF_TEXT.replace("&FCI NORB=   2,NELEC= 2,MS2=0,",
                                    "&FCI NORB=2, NELEC=2, MS2=0,")
        notation, reason = detect_fcidump_notation(plain)
        assert notation == "physicist"
        assert "standard" in reason

    def test_an_explicit_notation_beats_the_fingerprint(self):
        """A file that says what it is must be believed over a heuristic."""
        text = _PYSCF_TEXT.replace("  ISYM=1,\n", "  ISYM=1,\n  NOTATION=physicist,\n")
        notation, reason = detect_fcidump_notation(text)
        assert notation == "physicist"
        assert "NOTATION" in reason

    def test_the_writer_annotates_and_both_notations_round_trip(self):
        h1 = np.array([[-1.25, 0.0], [0.0, -0.6]])
        raw = np.zeros((2, 2, 2, 2))
        raw[0, 0, 0, 0] = 1.0
        raw[1, 1, 0, 0] = 0.5
        raw[1, 1, 1, 1] = 0.75
        raw[0, 1, 0, 1] = 0.25
        h2 = _symmetrise(raw)

        for notation in ("physicist", "chemist"):
            text = write_fcidump_text((h1, h2, 0.7), n_elec=2, notation=notation)
            assert f"NOTATION={notation}" in text
            back = read_fcidump_text(text)
            assert back.fcidump_notation == notation
            assert np.allclose(back.h2, h2)
            assert np.allclose(back.h1, h1)

    def test_annotate_false_keeps_a_standard_header_and_still_round_trips(self):
        """Without the marker the header itself must still say which it is.

        A physicist-labelled file is not PySCF output, so it must not carry
        PySCF's writer fingerprint.  With the old byte-identical header the
        detector called our own physicist file "chemist" and read every
        two-electron integral back transposed -- h2[1,1,0,0] came back as
        0.0625 instead of 0.25, silently.
        """
        h1 = np.array([[-1.25, 0.0], [0.0, -0.6]])
        raw = np.zeros((2, 2, 2, 2))
        raw[0, 0, 0, 0] = 1.0
        raw[1, 1, 0, 0] = 0.25
        raw[1, 1, 1, 1] = 0.75
        h2 = _symmetrise(raw)

        for notation in ("physicist", "chemist"):
            text = write_fcidump_text(
                (h1, h2, 0.0), n_elec=2, notation=notation, annotate=False
            )
            assert "NOTATION" not in text
            detected, _reason = detect_fcidump_notation(text)
            assert detected == notation, f"{notation} file detected as {detected}"
            if notation == "chemist":
                # A chemist file without the marker is indistinguishable from a
                # legacy Superfermion file, so the warning is expected here.
                with pytest.warns(UserWarning, match="0.1.25"):
                    back = read_fcidump_text(text)
            else:
                back = read_fcidump_text(text)
            assert back.fcidump_notation == notation
            assert np.allclose(back.h2, h2)
            assert np.allclose(back.h1, h1)

    def test_the_notation_argument_is_case_insensitive(self):
        """"AUTO" used to be rejected by an error saying "auto" is valid."""
        with pytest.warns(UserWarning, match="0.1.25"):
            assert read_fcidump_text(
                _PYSCF_TEXT, notation="AUTO"
            ).fcidump_notation == ("chemist")
        assert read_fcidump_text(
            _PYSCF_TEXT, notation="Physicist"
        ).fcidump_notation == "physicist"
        with pytest.raises(ValueError, match="'physicist', 'chemist' or 'auto'"):
            read_fcidump_text(_PYSCF_TEXT, notation="nonsense")

    def test_contradicting_the_marker_warns(self):
        text = _PYSCF_TEXT.replace("  ISYM=1,\n", "  ISYM=1,\n  NOTATION=chemist,\n")
        with pytest.warns(UserWarning, match="marked 'chemist'"):
            read_fcidump_text(text, notation="physicist")


class TestLegacySuperfermionFiles:
    """A file written by <= 0.1.25 carries PySCF's header, not PySCF's order.

    ``_PYSCF_TEXT`` above is byte-identical to what the 0.1.25 wheel emitted for
    ``write_fcidump_text((h1, h2, 0.0), n_elec=2)`` -- measured by writing the
    file with the 0.1.25 wheel and diffing the header.  Detecting ``chemist``
    from that header is right for PySCF and wrong for those files, and nothing
    in the text distinguishes them, so the reader must warn and the caller must
    be able to force the reading.
    """

    def test_the_legacy_header_warns_and_names_the_override(self):
        with pytest.warns(UserWarning, match="notation='physicist'"):
            mol = read_fcidump_text(_PYSCF_TEXT)
        assert mol.fcidump_notation == "chemist"
        # The auto reading transposes the two-electron block.
        assert mol.h2[1, 1, 0, 0] == pytest.approx(0.5)
        assert mol.h2[1, 0, 1, 0] == pytest.approx(0.0)

    def test_forcing_physicist_recovers_the_written_values(self):
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            mol = read_fcidump_text(_PYSCF_TEXT, notation="physicist")
        assert [str(w.message) for w in caught] == []
        assert mol.fcidump_notation == "physicist"
        assert mol.h2[1, 1, 0, 0] == pytest.approx(0.0)
        assert mol.h2[1, 0, 1, 0] == pytest.approx(0.5)

    def test_a_marked_file_does_not_warn(self):
        import warnings as _warnings

        text = _PYSCF_TEXT.replace("  ISYM=1,\n", "  ISYM=1,\n  NOTATION=physicist,\n")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            mol = read_fcidump_text(text)
        assert [str(w.message) for w in caught] == []
        assert mol.fcidump_notation == "physicist"
        assert mol.h2[1, 0, 1, 0] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. BP+OSD: osd_order is a real search budget
# ═══════════════════════════════════════════════════════════════════════════

# Steane/Hamming parity check: rank 3, so n_free = 7 - 3 = 4.
_STEANE_H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.int32,
)


def _brute_force_weight(H: np.ndarray, syndrome: np.ndarray) -> int:
    """Least weight of any error vector with this syndrome, over all 2**n."""
    n = H.shape[1]
    best = n + 1
    for pattern in range(2**n):
        error = np.array([(pattern >> i) & 1 for i in range(n)], dtype=np.int32)
        if np.array_equal((H @ error) % 2, syndrome):
            best = min(best, int(error.sum()))
    return best


class TestOsdOrderIsALiveBudget:
    """``osd_order`` used to be inert; now it changes the answer."""

    @staticmethod
    def _patterns(n: int = 7, max_weight: int = 2):
        out = []
        for pattern in range(2**n):
            error = np.array([(pattern >> i) & 1 for i in range(n)], dtype=np.int32)
            if 0 < int(error.sum()) <= max_weight:
                out.append(error)
        return out

    def test_osd_one_is_a_minimum_weight_decoder(self):
        decoder = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=1)
        mismatches = []
        for error in self._patterns():
            syndrome = (_STEANE_H @ error) % 2
            correction = decoder.decode(syndrome)
            got = len(correction)
            want = _brute_force_weight(_STEANE_H, syndrome)
            if got != want:
                mismatches.append((tuple(int(x) for x in error), got, want))
        assert mismatches == [], f"{len(mismatches)} heavier-than-minimum repairs: {mismatches}"

    def test_osd_really_searches_at_order_zero(self):
        """The zero-order reconstruction can be heavier than the minimum."""
        zero = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=0)
        one = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=1)
        differences = 0
        for error in self._patterns():
            syndrome = (_STEANE_H @ error) % 2
            if len(zero.decode(syndrome)) != len(one.decode(syndrome)):
                differences += 1
        assert differences > 0, "osd_order=0 and =1 are identical: the parameter is inert"

    def test_osd_saturates_at_the_number_of_free_columns(self):
        """Beyond n_free the enumeration is exhaustive; more cannot help."""
        one = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=1)
        big = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=30)
        for error in self._patterns():
            syndrome = (_STEANE_H @ error) % 2
            assert sorted(one.decode(syndrome)) == sorted(big.decode(syndrome))

    def test_a_negative_order_disables_osd(self):
        """``osd_order=-1`` is the documented way to ask for plain BP."""
        decoder = BPOSD_Decoder(h_matrix=_STEANE_H, error_rate=0.05, osd_order=-1)
        syndrome = (_STEANE_H @ np.array([1, 0, 0, 0, 0, 0, 0])) % 2
        assert len(decoder.decode(syndrome)) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. AVAS: the target set lives in the minimal basis
# ═══════════════════════════════════════════════════════════════════════════

_WATER = "O 0 0 0; H 0 0 0.96; H 0 0.93 0.24"


class TestAvasUsesTheMinimalBasis:
    """Basis-independent selection, matching PySCF's ``mcscf.avas``."""

    @pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz"])
    @pytest.mark.timeout(300)
    def test_water_o_target_selects_five_orbitals_in_every_basis(self, basis):
        from superfermion.chemistry.avas import avas_active_space

        pyscf = pytest.importorskip("pyscf")
        from pyscf.mcscf import avas as pavas

        reference = pyscf.gto.M(atom=_WATER, basis=basis, verbose=0)
        mean_field = pyscf.scf.RHF(reference).run()
        ncas, nelecas, _, _, _ = pavas._kernel(
            pavas.AVAS(mean_field, "O", threshold=0.2, verbose=0)
        )

        md = molecule_from_geometry(_WATER, basis=basis, name="h2o")
        space = avas_active_space(md, target_atoms=["O"], threshold=0.2)
        n_elec, n_orb = space.active_orbitals

        # 0.1.24 kept 6 here for STO-3G (and more for larger bases) because it
        # projected the working-basis AOs instead of the minao functions.
        assert (n_elec, n_orb) == (nelecas, ncas)
        assert n_orb == 5

    @pytest.mark.timeout(300)
    def test_the_projector_trace_equals_the_target_dimension(self):
        from superfermion.chemistry.avas import projector_metric

        pytest.importorskip("pyscf")
        md = molecule_from_geometry(_WATER, basis="sto-3g", name="h2o")
        oxygen = [
            i for i, label in enumerate(md.ao_labels) if label.split()[1] == "O"
        ]
        metric = projector_metric(md._mo_coeffs, md._ao_overlap, oxygen)
        assert float(np.trace(metric)) == pytest.approx(len(oxygen), abs=1e-9)
        assert metric.shape == (md.n_orbitals, md.n_orbitals)


# ═══════════════════════════════════════════════════════════════════════════
# 5. XYZ parsing
# ═══════════════════════════════════════════════════════════════════════════

_WATER_XYZ = (
    "3\n"
    "water\n"
    "O 0.0000 0.0000 0.1173\n"
    "H 0.0000 0.7572 -0.4692\n"
    "H 0.0000 -0.7572 -0.4692\n"
)


class TestXyzParsing:
    """The count on line 1 is now authoritative."""

    def test_a_standard_file_parses(self):
        title, geometry = parse_xyz_text(_WATER_XYZ, source="water.xyz")
        assert title == "water"
        assert geometry == (
            "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"
        )

    def test_extended_columns_are_dropped_not_forwarded(self):
        extended = (
            "3\ncharges in column 5\n"
            "O 0.0 0.0 0.1 -0.834\n"
            "H 0.0 0.7 -0.4 0.417 0.0 0.0 0.0\n"
            "H 0.0 -0.7 -0.4 0.417\n"
        )
        _, geometry = parse_xyz_text(extended)
        assert geometry == "O 0.0 0.0 0.1; H 0.0 0.7 -0.4; H 0.0 -0.7 -0.4"

    def test_crlf_and_a_leading_blank_line_are_tolerated(self):
        text = "\r\n2\r\ntitle\r\nH 0 0 0\r\nH 0 0 0.74\r\n"
        title, geometry = parse_xyz_text(text)
        assert title == "title"
        assert geometry == "H 0 0 0; H 0 0 0.74"

    def test_a_multi_frame_trajectory_reads_the_first_frame_and_warns(self):
        with pytest.warns(UserWarning, match="first frame"):
            _, geometry = parse_xyz_text(_WATER_XYZ * 2, source="traj.xyz")
        assert geometry.count(";") == 2

    @pytest.mark.parametrize(
        "text, message",
        [
            ("   \n\n", "the file is empty"),
            ("ATOMS\nwater\nH 0 0 0\n", "must be the atom count"),
            ("0\nnothing\n", "must be at least 1"),
            ("2\n", "ends before the title line"),
            ("3\nshort\nH 0 0 0\nH 0 0 0.74\n", "only 2 non-blank line"),
            ("1\nx\nH 0 0\n", r"expected 'symbol x y z'"),
            ("1\nx\nH 0 0 x\n", "are not numbers"),
            ("1\nx\n# 0 0 0\n", "is not an element symbol"),
        ],
    )
    def test_malformed_files_name_the_line_and_the_reason(self, text, message):
        with pytest.raises(ValueError, match=message):
            parse_xyz_text(text, source="bad.xyz")

    @pytest.mark.timeout(300)
    def test_molecule_from_xyz_builds_the_molecule(self, tmp_path):
        pytest.importorskip("pyscf")
        path = tmp_path / "water.xyz"
        path.write_text(_WATER_XYZ, encoding="utf-8")
        md = molecule_from_xyz(path, basis="sto-3g")
        assert md.name == "water"
        assert md.n_orbitals == 7
        assert len(md.geometry.split(";")) == 3

    @pytest.mark.timeout(300)
    def test_a_missing_title_falls_back_to_the_hill_formula(self, tmp_path):
        pytest.importorskip("pyscf")
        path = tmp_path / "no_title.xyz"
        path.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        assert molecule_from_xyz(path, basis="sto-3g").name == "H2"


# ═══════════════════════════════════════════════════════════════════════════
# 6. SCF convergence flag
# ═══════════════════════════════════════════════════════════════════════════

class TestScfConvergenceFlag:
    """``MolecularData.converged`` is queryable, not just a warning."""

    @pytest.mark.timeout(300)
    def test_a_converged_scf_records_true(self):
        pytest.importorskip("pyscf")
        md = molecule_from_geometry("H 0 0 0; H 0 0 0.74", basis="sto-3g")
        assert md.converged is True

    @pytest.mark.timeout(300)
    def test_a_stalled_scf_records_false_and_warns(self, monkeypatch):
        pytest.importorskip("pyscf")
        import pyscf.scf

        # ``pyscf.scf.RHF`` is a wrapper *function* (``pyscf.scf.hf.RHF`` is the
        # class), so the stand-in has to wrap the factory rather than subclass
        # it -- ``class X(pyscf.scf.RHF)`` raises TypeError.
        real_rhf = pyscf.scf.RHF

        def stalled_rhf(*args, **kwargs):
            mean_field = real_rhf(*args, **kwargs)
            kernel = mean_field.kernel

            def _kernel(*call_args, **call_kwargs):
                result = kernel(*call_args, **call_kwargs)
                # Exactly what PySCF does when it hits max_cycle.
                mean_field.converged = False
                return result

            mean_field.kernel = _kernel
            return mean_field

        monkeypatch.setattr(pyscf.scf, "RHF", stalled_rhf)
        with pytest.warns(UserWarning, match="did not converge"):
            md = molecule_from_geometry("H 0 0 0; H 0 0 0.74", basis="sto-3g")
        assert md.converged is False

    def test_the_default_is_optimistic_only_without_an_scf(self):
        """A hand-built record has no SCF to be wrong about."""
        record = MolecularData(name="x", geometry="H 0 0 0")
        assert record.converged is True


# ═══════════════════════════════════════════════════════════════════════════
# 7. AEGISS fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestAegissDoesNotGiveUp:
    """An over-aggressive entropy threshold returns a space, not an exception."""

    @pytest.mark.timeout(600)
    def test_water_with_the_o_target_returns_the_avas_space(self):
        pytest.importorskip("pyscf")
        from superfermion.chemistry.aegiss import aegiss_selection

        md = molecule_from_geometry(_WATER, basis="sto-3g", name="h2o")
        with pytest.warns(UserWarning, match="refinement_applied=False"):
            result = aegiss_selection(md, target_atoms=["O"])

        assert result["refinement_applied"] is False
        assert result["pruned"] == []
        assert result["n_active"] == 5
        assert len(result["active_indices"]) == result["n_active"]
        # The refinement was skipped, so it cannot have changed the energy.
        assert result["energy_pruned"] == pytest.approx(result["energy_avas"], abs=1e-12)

    @pytest.mark.timeout(600)
    def test_a_wider_window_actually_refines(self):
        pytest.importorskip("pyscf")
        from superfermion.chemistry.aegiss import aegiss_selection

        md = molecule_from_geometry(_WATER, basis="sto-3g", name="h2o")
        result = aegiss_selection(
            md, target_atoms=["O"], avas_threshold=0.1, entropy_threshold=0.01
        )
        assert result["refinement_applied"] is True
        assert 0 < result["n_active"] < result["avas"]["n_active"]
        assert result["energy_pruned"] > result["energy_avas"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. MPS bond dimension: 0 is not "unlimited"
# ═══════════════════════════════════════════════════════════════════════════


def _ghz3():
    circuit = sf.Circuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return circuit


class TestBondDimensionIsValidated:
    """``bond_dim=0`` sampled the uniform distribution instead of the circuit."""

    def test_zero_and_negative_are_refused_with_the_reason(self):
        from superfermion._limits import validate_bond_dim

        assert validate_bond_dim(64) == 64
        assert validate_bond_dim("8") == 8
        for bad in (0, -1, -64):
            with pytest.raises(ValueError, match="at least 1"):
                validate_bond_dim(bad)
        with pytest.raises(ValueError, match="positive integer"):
            validate_bond_dim(None)

    def test_a_run_with_bond_dim_zero_raises_instead_of_sampling_noise(self):
        # Measured before the guard: all eight GHZ-3 outcomes at ~25 counts each
        # out of 200 shots, i.e. a uniform distribution, with no error naming the
        # argument.  A negative value surfaced as pyo3's OverflowError instead.
        with pytest.raises(ValueError, match="bond_dim must be at least 1"):
            sf.run(_ghz3(), method="mps", shots=200, bond_dim=0, seed=7)
        with pytest.raises(ValueError, match="bond_dim must be at least 1"):
            sf.run(_ghz3(), method="mps", shots=200, bond_dim=-1, seed=7)

    def test_a_valid_bond_dimension_still_samples_the_state(self):
        counts = sf.run(_ghz3(), method="mps", shots=200, bond_dim=4, seed=7).counts
        assert set(counts) <= {"000", "111"}
        assert sum(counts.values()) == 200

    def test_the_observable_path_validates_the_same_way(self):
        from superfermion.observables.core import mps_expval

        with pytest.raises(ValueError, match="bond_dim must be at least 1"):
            mps_expval(_ghz3(), "ZZZ", bond_dim=0)

    def test_fractional_and_bool_values_are_refused(self):
        """``int()`` would truncate 3.7 to 3 and accept ``True`` as 1."""
        from superfermion._limits import validate_bond_dim

        with pytest.raises(ValueError, match="whole number"):
            validate_bond_dim(3.7)
        with pytest.raises(ValueError, match="bool is an int subclass"):
            validate_bond_dim(True)
        # An integral float, string or numpy scalar still names a real cap.
        assert validate_bond_dim(64.0) == 64
        assert validate_bond_dim(np.float64(64.0)) == 64
        assert validate_bond_dim(np.int64(16)) == 16
