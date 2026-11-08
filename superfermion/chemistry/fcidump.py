"""FCIDUMP and HDF5 ingest/egress for molecular integrals.

``MolecularData`` so far only existed if PySCF was installed and the SCF could
be re-run.  FCIDUMP breaks that dependency: it is a plain-text carrier for the
integrals of a *single* orbital set, so a calculation done elsewhere (PySCF,
Psi4, Molpro, Gaussian via a converter, or a previous run) can be resumed
without repeating the SCF.

Notation
--------
This is the one thing that silently ruins an import.  ``MolecularData.h2`` is
stored in **chemist** order, ``h2[p, q, r, s] == (pq|rs)``.  The labelling that
FCIDUMP files are actually written with is **also** chemist, although the
original description of the format is usually read as physicist ``<ij|kl>``;
the two are the same tensor with the middle indices swapped::

    (pq|rs)_chemist  ==  <pr|qs>_physicist

so reading one labelling as the other needs ``h2[p,q,r,s] = raw[p,r,s,q]`` and
not the identity.  Getting this backwards still produces a Hermitian operator
with plausible energies; only a comparison against an independent solver
catches it.  Both conventions are therefore selectable.

Every writer that could be checked writes chemist.  PySCF's ``tools.fcidump``
writes the chemist tensor ``(ij|kl)`` under the labels ``i j k l``, and so does
Qiskit Nature; NECI's manual states that "the FCIDUMP file that transfers the
electronic integrals from Molpro or Molcas to NECI assumes the chemist's
notation", HANDE's format notes say the Coulomb integrals are chemist's, and
HORTON and orbkit (whose format "is adopted from Molpro") do the same.  Reading
a PySCF-written LiH file with the physicist labelling gives -10.543477 Ha
instead of the correct -7.880982 Ha, with no error; in the other direction a
physicist-labelled H2/STO-3G file gives -1.967656 Ha in Qiskit Nature against
-1.137306 Ha for the chemist one.  :func:`write_fcidump_text` therefore
*labels* chemist by default, so its output lands in either reader untransposed;
pass ``notation="physicist"`` for the literal standard, which is what a Molpro
``{FCI,DUMP=}`` file is *not*.

``notation="auto"`` (the default) therefore decides from the file:

1. a ``NOTATION=`` variable in the ``&FCI`` header wins -- every file written
   by :func:`write_fcidump` carries one, so round trips are exact;
2. otherwise Superfermion 0.1.26-0.1.27's physicist header -- those two
   releases put a space before the closing comma (``MS2=0 ,``) so that their
   physicist output would stay out of PySCF's fingerprint -- is read as
   ``"physicist"``;
3. otherwise Qiskit Nature's ``to_file`` header or PySCF's writer fingerprint
   (``' &FCI NORB=%4d,NELEC=%2d,MS2=%d,'``, commas butted against the values)
   is read as ``"chemist"``, which is what those writers wrote;
4. otherwise ``"chemist"``, the convention shared by every writer above.  A
   file with no marker and no known fingerprint used to fall back to the
   physicist labelling, which transposed Molpro, Molcas, Psi4, HORTON and
   orbkit output -- none of which carry a fingerprint -- silently.

The fingerprint is not exclusive to PySCF.  Superfermion 0.1.25 and earlier
wrote the same ``&FCI NORB=%4d,NELEC=%2d,MS2=%d,`` header, and the same
``ORBSYM``/``ISYM`` lines, for physicist-ordered output; measured against
PySCF 2.14.0 the two headers are byte for byte identical, so no inspection of
the text can separate them.  Rule 3 therefore runs with a warning naming the
override: pass ``notation="physicist"`` for a pre-0.1.26 Superfermion file.
The writer annotates every file, so rule 1 answers for superfermion's own
output and no warning is raised; ``annotate=False`` opts out of the marker, and
a chemist file written that way -- the default layout from 0.1.28 -- is still
recognised by the fingerprint (with the same warning, since the text alone
cannot rule out a legacy file).

Exponent fields are deliberately *not* part of the fingerprint.  The obvious
idea -- "PySCF prints ``%.16g``, so a Fortran ``E``/``D`` field means some other
writer" -- is wrong on real files: on water/STO-3G, 128 of the 547 lines of a
file written by ``pyscf.tools.fcidump.from_mo`` carry values such as
``4.040691484975394e-05``, because ``%g`` switches to exponent form for small
magnitudes.  Keying on that turned a genuine PySCF file into ``"physicist"``
and reintroduced the very transposition this detection exists to avoid.

The choice is recorded on the result as ``MolecularData.fcidump_notation`` so
it can be checked instead of assumed.  Pass ``notation=`` explicitly to
override the detection (e.g. for a Molpro file that also lacks a marker).

Usage:
    >>> from superfermion.chemistry.fcidump import read_fcidump, write_fcidump
    >>> mol = read_fcidump("h2o.fcidump", geometry="O 0 0 0; H 0 0 1; H 0 1 0")
    >>> mol.fcidump_notation  # 'physicist' or 'chemist'
    >>> H = mol.to_hamiltonian(active_space=(6, 6))
    >>> write_fcidump(mol, "copy.fcidump")
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from superfermion.chemistry.pyscf_bridge import MolecularData

__all__ = [
    "read_fcidump",
    "write_fcidump",
    "read_fcidump_text",
    "write_fcidump_text",
    "detect_fcidump_notation",
    "to_hdf5",
    "from_hdf5",
]

_DEFAULT_FLOAT_FORMAT = "%22.16f"

# ``NOTATION=physicist`` or ``NOTATION=chemist`` inside the &FCI namelist.
# Extra namelist variables are skipped by token-scanning readers (PySCF's and
# Qiskit Nature's included), and a file that carries it is read exactly as
# written, so a PySCF-vs-standard guess is never needed for superfermion's own
# output.
_NOTATION_RE = re.compile(r"\bNOTATION\s*=\s*[\"']?(physicist|chemist)\b", re.IGNORECASE)

# PySCF's ``tools.fcidump.write_head`` emits
# ``' &FCI NORB=%4d,NELEC=%2d,MS2=%d,'`` -- note the commas butted directly
# against the values, and the space padding that ``%4d``/``%2d`` leave behind.
# A hand-written or Molpro-style header writes ``NORB=7, NELEC=2`` instead, so
# this is a writer fingerprint rather than a format requirement.
#
# Exponent fields are NOT evidence against PySCF: ``write_eri`` uses ``%.16g``,
# which prints ``4.040691484975394e-05`` for small integrals (measured: 128 of
# 547 lines on water/STO-3G).  An earlier version of this detector vetoed the
# fingerprint on any ``E``/``D`` exponent and therefore mis-read every PySCF
# file that happened to contain a small integral -- which is most of them.
_PYSCF_HEADER_RE = re.compile(r"&\s*FCI\s+NORB=\s*\d+,NELEC=\s*\d+,MS2=-?\d+,")

# Superfermion 0.1.26 and 0.1.27 wrote physicist output with a space before the
# closing comma -- ``MS2=0 ,`` -- so that it would drop out of PySCF's
# fingerprint, and chemist output with PySCF's exact bytes.  Measured from the
# tagged sources (``header_end = "," if notation == "chemist" else " ,"``).
# That space is the only thing separating an unannotated physicist file from
# those two releases from every other writer, so it is matched explicitly:
# without it those files would now fall through to the chemist default below.
_SUPERFERMION_PHYSICIST_RE = re.compile(
    r"&\s*FCI\s+NORB=\s*\d+,NELEC=\s*\d+,MS2=-?\d+\s+,"
)

# Qiskit Nature's ``FCIDump.to_file`` writes
# ``f"&FCI NORB={norb:4d},NELEC={nelec:4d},MS2={ms2:4d},"`` -- measured on
# qiskit-nature 0.7.2.  Three things separate it from PySCF's header: the line
# starts in column 0 rather than with a space, ``NELEC`` is padded to four
# columns rather than two, and ``MS2`` is padded rather than bare.  Every field
# therefore keeps at least one space, which is what the ``+`` below asserts;
# with ``*`` instead this regex also matched HORTON's unpadded
# ``&FCI NORB=2,NELEC=2,MS2=0,`` and stole its PySCF-fingerprint warning.
# It matters because ``to_file`` rebuilds the header from Qiskit's own
# dataclass and so drops any ``NOTATION=`` marker, leaving a chemist file that
# carries no marker of its own.
_QISKIT_HEADER_RE = re.compile(
    r"^&FCI NORB= +\d+,NELEC= +\d+,MS2= +-?\d+,\s*$", re.MULTILINE
)

# The three transpositions that generate the integral's eight-fold symmetry.
# Equality under each of them implies equality under all eight, so these are
# enough to decide whether a tensor can be described by the canonical quarter
# that :func:`write_fcidump_text` writes.
_SYMMETRY_CHECKS: Tuple[Tuple[Tuple[int, int, int, int], str], ...] = (
    ((1, 0, 2, 3), "(pq|rs) != (qp|rs)"),
    ((0, 1, 3, 2), "(pq|rs) != (pq|sr)"),
    ((2, 3, 0, 1), "(pq|rs) != (rs|pq)"),
)


def _symmetry_violations(h2: np.ndarray) -> List[Tuple[str, float]]:
    """Return ``(description, max deviation)`` for each broken symmetry.

    Walks one leading index at a time so the comparison never materialises a
    second full ``norb``-to-the-fourth array.
    """
    norb = h2.shape[0]
    violations: List[Tuple[str, float]] = []
    for order, message in _SYMMETRY_CHECKS:
        image = h2.transpose(order)
        deviation = 0.0
        for i in range(norb):
            block = np.abs(h2[i] - image[i])
            if block.size:
                deviation = max(deviation, float(block.max()))
        if deviation > 0.0:
            violations.append((message, deviation))
    return violations


def detect_fcidump_notation(text: str) -> Tuple[str, str]:
    """Return ``(notation, reason)`` for an FCIDUMP body.

    ``notation`` is ``"physicist"`` or ``"chemist"``; ``reason`` is a short
    human-readable explanation of the evidence used.

    The default for a file that carries no evidence either way is **chemist**,
    because that is the convention of every FCIDUMP writer that could be
    documented: NECI's manual states that "the FCIDUMP file that transfers the
    electronic integrals from Molpro or Molcas to NECI assumes the chemist's
    notation", HANDE's format notes say the Coulomb integrals are chemist's,
    HORTON's manual and orbkit (whose format "is adopted from Molpro") say the
    same, and PySCF and Qiskit Nature both write it.  Reading an unmarked file
    as the physicist labelling transposed every one of those by default.

    ``"chemist"`` from the PySCF fingerprint means *PySCF-shaped*, not *known
    to be PySCF*: Superfermion 0.1.25 and earlier emitted the same header for
    physicist-ordered data.  :func:`read_fcidump_text` warns when it acts on
    that rule alone, because a pre-0.1.26 Superfermion file needs
    ``notation="physicist"`` to be read correctly.
    """
    match = _NOTATION_RE.search(text)
    if match is not None:
        return match.group(1).lower(), "NOTATION= marker in the &FCI header"
    if _SUPERFERMION_PHYSICIST_RE.search(text):
        return (
            "physicist",
            "Superfermion 0.1.26-0.1.27 physicist header (space before the comma)",
        )
    if _QISKIT_HEADER_RE.search(text):
        return "chemist", "Qiskit Nature writer fingerprint"
    if _PYSCF_HEADER_RE.search(text):
        return "chemist", "PySCF writer fingerprint"
    return "chemist", "FCIDUMP chemist convention (no writer marker)"


def _parse_header(text: str) -> Dict[str, Any]:
    """Parse the ``&FCI ... &END`` block into a dict of ints/lists."""
    match = re.search(r"&FCI(.*?)&END", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError("no &FCI ... &END header found; not an FCIDUMP file")

    body = match.group(1).replace("/", "")
    body = re.sub(r"\s+", " ", body).strip().strip(",")
    # Newline was a legal separator in the header; collapse before splitting
    # so "NORB=6\n NELEC=6" does not become one token.
    body = body.replace(" NORB", ",NORB").replace(" NELEC", ",NELEC")
    body = body.replace(" MS2", ",MS2").replace(" MS", ",MS")
    body = body.replace(" ISYM", ",ISYM").replace(" ORBSYM", ",ORBSYM")

    fields: Dict[str, Any] = {}
    for token in re.split(r",(?=[A-Za-z_]+\s*=)", body):
        token = token.strip().strip(",")
        if not token or "=" not in token:
            continue
        key, _, raw = token.partition("=")
        key = key.strip().upper()
        values = [v for v in raw.replace(",", " ").split() if v]
        if not values:
            continue
        if key == "ORBSYM":
            fields[key] = [int(float(v)) for v in values]
        else:
            try:
                fields[key] = int(float(values[0]))
            except ValueError:
                # Non-numeric namelist variables (``NOTATION=physicist``) are
                # metadata, not counts; keep them as text instead of failing.
                fields[key] = "".join(values)
    return fields


def _to_chemist(
    raw: Dict[Tuple[int, int, int, int], float], norb: int, notation: str
) -> np.ndarray:
    """Expand canonical physicist/chemist ERI entries to a chemist array."""
    h2 = np.zeros((norb, norb, norb, norb), dtype=float)
    for (i, j, k, l), value in raw.items():
        if notation == "chemist":
            p, q, r, s = i, j, k, l
        elif notation == "physicist":
            # <ij|kl> = (ik|jl): undo the middle swap when storing.
            p, q, r, s = i, k, j, l
        else:
            raise ValueError(
                f"notation must be 'physicist', 'chemist' or 'auto', got {notation!r}"
            )
        h2[p, q, r, s] = value
        h2[q, p, r, s] = value
        h2[p, q, s, r] = value
        h2[q, p, s, r] = value
        h2[r, s, p, q] = value
        h2[s, r, p, q] = value
        h2[r, s, q, p] = value
        h2[s, r, q, p] = value
    return h2


def read_fcidump_text(
    text: str,
    *,
    notation: str = "auto",
    geometry: Optional[str] = None,
    basis: str = "fcidump",
    name: Optional[str] = None,
    charge: int = 0,
    tol: float = 0.0,
) -> MolecularData:
    """Parse FCIDUMP text into a :class:`MolecularData`.

    Args:
        text: Full file contents.
        notation: ``"auto"`` (default) decides from the file itself -- see
            :func:`detect_fcidump_notation`; ``"physicist"`` and
            ``"chemist"`` force the labelling.
        geometry: Optional geometry string.  FCIDUMP carries no geometry, so
            this is only metadata unless you also want to re-run SCF.
        basis: Basis label recorded on the result.  Defaults to ``"fcidump"``
            because the true basis is not stored in the file.
        name: Molecule name; defaults to ``"fcidump"``.
        charge: Recorded molecular charge.
        tol: Drop integral values with ``|value| <= tol``.

    Returns:
        ``MolecularData`` with ``h1``, ``h2`` (chemist), ``nuclear_repulsion``
        set to the FCIDUMP core energy, ``n_orbitals = NORB``,
        ``n_electrons = NELEC``, ``fcidump_notation`` set to the convention
        that was used, and ``fcidump_orbsym``/``fcidump_isym`` carried over
        from the header so a write-back does not drop them.
    """
    notation = "auto" if notation is None else str(notation).strip().lower()
    if notation not in ("auto", "physicist", "chemist"):
        raise ValueError(
            f"notation must be 'physicist', 'chemist' or 'auto', got {notation!r}"
        )
    if notation == "auto":
        notation, reason = detect_fcidump_notation(text)
        if reason.startswith("PySCF writer fingerprint"):
            # The header alone cannot separate a PySCF file from a Superfermion
            # <= 0.1.25 file: both printed the same &FCI line, and the second
            # one holds physicist-ordered data that this rule transposes.
            warnings.warn(
                "read as 'chemist' from PySCF's header fingerprint, which "
                "Superfermion 0.1.25 and earlier also wrote for "
                "physicist-ordered files.  If this file came from an older "
                "Superfermion, pass notation='physicist'; otherwise every "
                "two-electron integral is transposed.  Pass notation "
                "explicitly to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
    else:
        marked, mark_reason = detect_fcidump_notation(text)
        if mark_reason.startswith("NOTATION") and marked != notation:
            warnings.warn(
                f"the file is marked {marked!r} but was read as {notation!r}; "
                "the explicit argument wins, so the integrals may be wrong",
                UserWarning,
                stacklevel=2,
            )
    header = _parse_header(text)
    norb = int(header["NORB"])
    nelec = int(header["NELEC"])
    ms2 = int(header.get("MS2", header.get("MS", 0)))
    # Carry the symmetry labels only when they line up with NORB.  A file that
    # states the wrong count is malformed; dropping the labels there keeps the
    # write-back working exactly as it did before they were kept at all.
    orbsym: Optional[Tuple[int, ...]] = None
    if "ORBSYM" in header:
        stated = tuple(int(x) for x in header["ORBSYM"])
        orbsym = stated if len(stated) == norb else None
        # HANDE's format notes define ``ORBSYM(i) = S + 1`` for the irrep an
        # orbital spans, with 0 meaning "symmetry not well defined", so 0..8 is
        # the whole legal range; PySCF's ``to_scf`` raises outright on a
        # negative label.  Anything outside it is a writer bug or a different
        # convention, and is worth saying out loud -- but the labels are not
        # remapped, because PySCF and Molpro order the D2h irreps differently
        # (``pyscf.tools.fcidump.ORBSYM_MAP``) and guessing which one a file
        # used is not possible from the text.
        if orbsym is not None:
            outside = [label for label in orbsym if not 0 <= label <= 8]
            if outside:
                warnings.warn(
                    f"ORBSYM states {len(outside)} label(s) outside the 0..8 "
                    f"range the format defines (first: {outside[0]}).  They "
                    "are carried through verbatim, but a symmetry-adapted "
                    "reader may reject the file.",
                    UserWarning,
                    stacklevel=2,
                )
    isym = int(header.get("ISYM", 1))

    body = re.sub(r"&FCI.*?&END", "", text, flags=re.IGNORECASE | re.DOTALL)

    h1 = np.zeros((norb, norb), dtype=float)
    raw_h2: Dict[Tuple[int, int, int, int], float] = {}
    ecore = 0.0

    line_re = re.compile(
        r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    for match in line_re.finditer(body):
        value = float(match.group(1).replace("d", "e").replace("D", "e"))
        i, j, k, l = (int(match.group(n)) for n in (2, 3, 4, 5))
        if abs(value) <= tol:
            continue
        if k == 0:
            if j == 0:
                ecore = value
            else:
                h1[i - 1, j - 1] = value
                h1[j - 1, i - 1] = value
        else:
            raw_h2[(i - 1, j - 1, k - 1, l - 1)] = value

    h2 = _to_chemist(raw_h2, norb, notation)

    return MolecularData(
        name=name or "fcidump",
        geometry=geometry or "",
        basis=basis,
        charge=charge,
        spin=ms2,
        h1=h1,
        h2=h2,
        nuclear_repulsion=ecore,
        hf_energy=0.0,
        n_orbitals=norb,
        n_electrons=nelec,
        fcidump_notation=notation,
        fcidump_orbsym=orbsym,
        fcidump_isym=isym,
    )


def read_fcidump(
    path: str, **kwargs: Any
) -> MolecularData:
    """Read an FCIDUMP file.  See :func:`read_fcidump_text` for keywords."""
    with open(path, "r", encoding="utf-8") as handle:
        return read_fcidump_text(handle.read(), **kwargs)


def write_fcidump_text(
    spec: Union[MolecularData, Tuple[np.ndarray, np.ndarray, float]],
    *,
    n_elec: Optional[int] = None,
    spin: Optional[int] = None,
    orbsym: Optional[Sequence[int]] = None,
    isym: Optional[int] = None,
    notation: str = "chemist",
    tol: float = 1e-12,
    float_format: str = _DEFAULT_FLOAT_FORMAT,
    annotate: bool = True,
    validate: bool = True,
) -> str:
    """Serialise integrals to FCIDUMP text.

    One record is written per symmetry orbit of the internal chemist tensor,
    selected on the tensor indices (``i >= j``, ``k >= l``, pair ``kl`` no later
    than pair ``ij``), and a reader that regenerates the eight-fold symmetry
    family from each record reconstructs the tensor exactly.  For chemist
    output those indices are also the labels on disk, matching the record order
    PySCF's ``symm.pack`` walks; for physicist output the labels are
    ``i k j l``, so the records are canonical in the tensor rather than in
    label space.

    Args:
        spec: ``MolecularData`` or ``(h1, h2, ecore)``.  ``h2`` must be in
            chemist order, matching ``MolecularData.h2``.
        n_elec: Electron count; defaults to ``spec.n_electrons``.
        spin: 2S written as ``MS2``; defaults to ``spec.spin``.
        orbsym: Orbital symmetry labels, one D2h irrep (1..8) per orbital.
            Defaults to the ``ORBSYM`` line the file carried when ``spec`` is
            a :class:`MolecularData` that :func:`read_fcidump` produced, and
            to all-ones otherwise.  The length must equal the orbital count.
        isym: State symmetry label.  Defaults to the value read from the same
            file, or 1.
        notation: How to label the entries on disk.  ``"chemist"`` (the
            default since 0.1.28) writes ``(pq|rs)`` as ``p q r s``, the
            layout PySCF's ``tools.fcidump`` and Qiskit Nature's ``FCIDump``
            read and write; ``"physicist"`` writes ``<pr|qs>`` under the same
            labels, which is the FCIDUMP standard those two readers do not
            implement.  A physicist-labelled file handed to them comes back
            transposed and silent: on water/STO-3G that is -124.099 Ha against
            -75.013 Ha.
        tol: Skip entries below this magnitude.
        float_format: ``printf`` format for the integral value.
        annotate: Record the convention as a ``NOTATION=`` variable in the
            ``&FCI`` header, so :func:`read_fcidump` reproduces the tensor
            exactly instead of inferring the layout.  PySCF and Qiskit Nature
            skip the variable and then apply their own (chemist) reading, so
            the marker protects superfermion's round trips, not theirs.  Note
            that both of those writers rebuild the header from scratch, so the
            variable does not survive a file they rewrite: PySCF's replacement
            header is caught by :func:`detect_fcidump_notation`'s fingerprint,
            but Qiskit Nature's is not (it pads ``MS2`` where PySCF does not),
            so a file that has been through ``FCIDump.to_file`` needs
            ``notation="chemist"`` when it is read back here.  With
            ``annotate=False`` there is no marker: a physicist-labelled file
            then carries a header that is *not* byte-identical to PySCF's (one
            space before the final comma), so :func:`detect_fcidump_notation`
            still recovers the right convention, while chemist output keeps
            PySCF's exact bytes and is recognised by the fingerprint -- with a
            warning, because those bytes could also come from 0.1.25.
        validate: Warn when ``h2`` is not invariant under the integral's
            eight-fold symmetry.  The writer emits one record per orbit and a
            reader fills the other seven images from it, so any part of ``h2``
            outside the canonical quarter is dropped rather than written; the
            warning names the deviation so an asymmetric tensor is not lost
            silently.  A violation counts when it exceeds ``max(tol, 1e-12 *
            max|h2|)``, which keeps the rounding in a real ``ao2mo`` tensor --
            measured 1.7e-16 on H2/STO-3G for the pair swap -- out of it.

    Returns:
        The file body as a string.
    """
    if isinstance(spec, MolecularData):
        h1, h2, ecore = spec.h1, spec.h2, spec.nuclear_repulsion
        n_elec = spec.n_electrons if n_elec is None else n_elec
        spin = spec.spin if spin is None else spin
        orbsym = spec.fcidump_orbsym if orbsym is None else orbsym
        isym = spec.fcidump_isym if isym is None else isym
    else:
        h1, h2, ecore = spec

    if h1 is None:
        raise ValueError("cannot write an FCIDUMP without one-body integrals")

    h1 = np.asarray(h1, dtype=float)
    norb = h1.shape[0]
    if n_elec is None:
        raise ValueError("n_elec is required when spec carries no MolecularData")
    if spin is None:
        spin = 0
    if isym is None:
        isym = 1
    # ``orbsym`` may be a numpy array, where ``orbsym or [...]`` raises on the
    # ambiguous truth value; test for None instead.
    labels = [1] * norb if orbsym is None else [int(x) for x in orbsym]
    if len(labels) != norb:
        raise ValueError(
            f"orbsym carries {len(labels)} labels for {norb} orbitals"
        )
    outside = [label for label in labels if not 0 <= label <= 8]
    if outside:
        # 0-based irrep ids are the usual mistake here, and they write a header
        # that reads back shifted rather than failing.  ``0`` is legal on its
        # own -- HANDE uses it for an orbital with no well defined symmetry.
        warnings.warn(
            f"orbsym carries {len(outside)} label(s) outside the 0..8 range "
            f"the format defines (first: {outside[0]}); 0-based irrep ids are "
            "the usual cause.  The labels are written as given, so a reader "
            "that interprets them will see different irreps.",
            UserWarning,
            stacklevel=2,
        )

    # PySCF's writer emits ``NORB=%4d,NELEC=%2d,MS2=%d,`` and the detector keys
    # on exactly that spacing, so output a reader could mistake for PySCF's has
    # to say chemist -- which is the default.  A physicist-labelled file
    # therefore leaves one space before the final comma: every field stays
    # comma-separated (PySCF splits on "," and "=", so the space is inert) while
    # the header drops out of the fingerprint, and the file re-detects as the
    # physicist file it is.  Chemist output keeps PySCF's exact bytes, because
    # there the labels and the header agree.
    header_end = "," if notation == "chemist" else " ,"
    lines = [
        f" &FCI NORB={norb:4d},NELEC={int(n_elec):2d},MS2={int(spin)}{header_end}",
        "  ORBSYM=" + ",".join(str(x) for x in labels) + ",",
        f"  ISYM={int(isym)},",
    ]
    if annotate:
        lines.append(f"  NOTATION={notation},")
    lines.append(" &END")

    if h2 is not None:
        h2 = np.asarray(h2, dtype=float)
        if notation not in ("physicist", "chemist"):
            raise ValueError(
                f"notation must be 'physicist' or 'chemist', got {notation!r}"
            )
        if h2.shape != (norb, norb, norb, norb):
            raise ValueError(
                f"h2 must have shape {(norb, norb, norb, norb)} for the {norb} "
                f"orbitals of h1, got {h2.shape}"
            )
        if validate:
            # Relative as well as absolute: a real ``ao2mo.restore`` tensor is
            # bitwise equal under (p<->q) and (r<->s) but the pair swap differs
            # by rounding -- 1.7e-16 on H2/STO-3G, and proportionally larger on
            # a tensor with core integrals.  A fixed 1e-12 floor would warn on
            # such a file, so the threshold scales with the largest element.
            scale = float(np.max(np.abs(h2))) if h2.size else 0.0
            threshold = max(tol, 1e-12 * scale)
            broken = [
                (message, deviation)
                for message, deviation in _symmetry_violations(h2)
                if deviation > threshold
            ]
            if broken:
                detail = "; ".join(
                    f"{message} by up to {deviation:.3e}"
                    for message, deviation in broken
                )
                warnings.warn(
                    f"h2 is not symmetric under the integral's eight-fold "
                    f"symmetry ({detail}).  One record per orbit is written "
                    "and every other image is filled from the canonical "
                    "member of that orbit, so the part of h2 outside the "
                    "canonical quarter is dropped.  Pass validate=False to "
                    "write without this check.",
                    UserWarning,
                    stacklevel=2,
                )
        # One record per symmetry orbit, selected on the *tensor* (``i >= j``,
        # ``k >= l``, pair ``kl`` no later than ``ij``), which is the quarter
        # PySCF's ``symm.pack`` walks and the one this reader expands back.
        # The selection has to match the expansion: choosing the quarter in
        # label space instead covers 50 of the 55 orbits for n = 4, because the
        # label permutation that ``notation="physicist"`` applies does not
        # normalise the integral's symmetry group, and the round trip then
        # leaves 24 of the 256 array elements at zero.
        pair = [(i, j) for i in range(norb) for j in range(i + 1)]
        for a, (i, j) in enumerate(pair):
            for b in range(a + 1):
                k, l = pair[b]
                if notation == "physicist":
                    # (ij|kl) == <ik|jl>, so the value goes under the labels i k j l
                    lab = (i + 1, k + 1, j + 1, l + 1)
                else:
                    lab = (i + 1, j + 1, k + 1, l + 1)
                value = h2[i, j, k, l]
                if abs(value) <= tol:
                    continue
                lines.append(float_format % value + " %4d %4d %4d %4d" % lab)
    elif notation not in ("physicist", "chemist"):
        raise ValueError(
            f"notation must be 'physicist' or 'chemist', got {notation!r}"
        )

    for i in range(norb):
        for j in range(i + 1):
            value = h1[i, j]
            if abs(value) <= tol:
                continue
            lines.append(float_format % value + " %4d %4d  0  0" % (i + 1, j + 1))

    lines.append(float_format % float(ecore) + "  0  0  0  0")
    return "\n".join(lines) + "\n"


def write_fcidump(
    spec: Union[MolecularData, Tuple[np.ndarray, np.ndarray, float]],
    path: str,
    **kwargs: Any,
) -> str:
    """Write an FCIDUMP file.  See :func:`write_fcidump_text` for keywords."""
    text = write_fcidump_text(spec, **kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def to_hdf5(mol: MolecularData, path: str, *, group: str = "integrals") -> str:
    """Store a :class:`MolecularData` in an HDF5 file.

    This is superfermion's own schema, not qiskit-nature's ``HDF5Driver``
    layout: one group holding ``h1``, ``h2``, the scalars, and a few strings.
    Round-tripping with superfermion is guaranteed; round-tripping with another
    code is not.

    Requires ``h5py``.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError(
            "HDF5 support needs h5py: pip install h5py"
        ) from exc

    with h5py.File(path, "w") as handle:
        grp = handle.create_group(group)
        if mol.h1 is not None:
            grp.create_dataset("h1", data=np.asarray(mol.h1, dtype=float))
        if mol.h2 is not None:
            grp.create_dataset("h2", data=np.asarray(mol.h2, dtype=float))
        grp.attrs["name"] = mol.name
        grp.attrs["geometry"] = mol.geometry
        grp.attrs["basis"] = mol.basis
        grp.attrs["charge"] = int(mol.charge)
        grp.attrs["spin"] = int(mol.spin)
        grp.attrs["nuclear_repulsion"] = float(mol.nuclear_repulsion)
        grp.attrs["hf_energy"] = float(mol.hf_energy)
        grp.attrs["n_orbitals"] = int(mol.n_orbitals)
        grp.attrs["n_electrons"] = int(mol.n_electrons)
    return path


def from_hdf5(path: str, *, group: str = "integrals") -> MolecularData:
    """Read a :class:`MolecularData` written by :func:`to_hdf5`."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError(
            "HDF5 support needs h5py: pip install h5py"
        ) from exc

    with h5py.File(path, "r") as handle:
        grp = handle[group]
        attrs = dict(grp.attrs)
        h1 = np.asarray(grp["h1"]) if "h1" in grp else None
        h2 = np.asarray(grp["h2"]) if "h2" in grp else None

    def _text(key: str, default: str = "") -> str:
        value = attrs.get(key, default)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    return MolecularData(
        name=_text("name", "hdf5"),
        geometry=_text("geometry"),
        basis=_text("basis", "hdf5"),
        charge=int(attrs.get("charge", 0)),
        spin=int(attrs.get("spin", 0)),
        h1=h1,
        h2=h2,
        nuclear_repulsion=float(attrs.get("nuclear_repulsion", 0.0)),
        hf_energy=float(attrs.get("hf_energy", 0.0)),
        n_orbitals=int(attrs.get("n_orbitals", h1.shape[0] if h1 is not None else 0)),
        n_electrons=int(attrs.get("n_electrons", 0)),
    )
