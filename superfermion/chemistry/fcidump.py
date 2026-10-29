"""FCIDUMP and HDF5 ingest/egress for molecular integrals.

``MolecularData`` so far only existed if PySCF was installed and the SCF could
be re-run.  FCIDUMP breaks that dependency: it is a plain-text carrier for the
integrals of a *single* orbital set, so a calculation done elsewhere (PySCF,
Psi4, Molpro, Gaussian via a converter, or a previous run) can be resumed
without repeating the SCF.

Notation
--------
This is the one thing that silently ruins an import.  ``MolecularData.h2`` is
stored in **chemist** order, ``h2[p, q, r, s] == (pq|rs)``.  The FCIDUMP
standard labels integrals in **physicist** order, ``<ij|kl>``, which is the
same tensor with the middle indices swapped::

    (pq|rs)_chemist  ==  <pr|qs>_physicist

so reading a standard file needs ``h2[p,q,r,s] = raw[p,r,s,q]`` and not the
identity.  Getting this backwards still produces a Hermitian operator with
plausible energies; only a comparison against an independent solver catches
it.  Both conventions are therefore selectable, ``notation="physicist"`` is
the default (the standard), and ``notation="chemist"`` is provided because
PySCF's own ``fcidump.from_integrals`` writes the chemist tensor under
unpermuted labels.  ``_verify_fcidump.py`` pins the choice against PySCF FCI.

Usage:
    >>> from superfermion.chemistry.fcidump import read_fcidump, write_fcidump
    >>> mol = read_fcidump("h2o.fcidump", geometry="O 0 0 0; H 0 0 1; H 0 1 0")
    >>> H = mol.to_hamiltonian(active_space=(6, 6))
    >>> write_fcidump(mol, "copy.fcidump")
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from superfermion.chemistry.pyscf_bridge import MolecularData

__all__ = [
    "read_fcidump",
    "write_fcidump",
    "read_fcidump_text",
    "write_fcidump_text",
    "to_hdf5",
    "from_hdf5",
]

_DEFAULT_FLOAT_FORMAT = "%22.16f"


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
            fields[key] = int(float(values[0]))
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
                f"notation must be 'physicist' or 'chemist', got {notation!r}"
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
    notation: str = "physicist",
    geometry: Optional[str] = None,
    basis: str = "fcidump",
    name: Optional[str] = None,
    charge: int = 0,
    tol: float = 0.0,
) -> MolecularData:
    """Parse FCIDUMP text into a :class:`MolecularData`.

    Args:
        text: Full file contents.
        notation: ``"physicist"`` (FCIDUMP standard, default) or ``"chemist"``
            (PySCF's writer).  See the module docstring.
        geometry: Optional geometry string.  FCIDUMP carries no geometry, so
            this is only metadata unless you also want to re-run SCF.
        basis: Basis label recorded on the result.  Defaults to ``"fcidump"``
            because the true basis is not stored in the file.
        name: Molecule name; defaults to ``"fcidump"``.
        charge: Recorded molecular charge.
        tol: Drop integral values with ``|value| <= tol``.

    Returns:
        ``MolecularData`` with ``h1``, ``h2`` (chemist), ``nuclear_repulsion``
        set to the FCIDUMP core energy, ``n_orbitals = NORB`` and
        ``n_electrons = NELEC``.
    """
    header = _parse_header(text)
    norb = int(header["NORB"])
    nelec = int(header["NELEC"])
    ms2 = int(header.get("MS2", header.get("MS", 0)))

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
    isym: int = 1,
    notation: str = "physicist",
    tol: float = 1e-12,
    float_format: str = _DEFAULT_FLOAT_FORMAT,
) -> str:
    """Serialise integrals to FCIDUMP text.

    Only the canonical entries are written (``i >= j``, ``k >= l``, ``ij >= kl``
    in *label* order), which is what PySCF and OpenFermion emit, so a reader
    that keeps only canonical entries reconstructs the full tensor.

    Args:
        spec: ``MolecularData`` or ``(h1, h2, ecore)``.  ``h2`` must be in
            chemist order, matching ``MolecularData.h2``.
        n_elec: Electron count; defaults to ``spec.n_electrons``.
        spin: 2S written as ``MS2``; defaults to ``spec.spin``.
        orbsym: Orbital symmetry labels; defaults to all-ones.
        isym: State symmetry label.
        notation: How to label the entries on disk.  ``"physicist"`` (default,
            the standard) writes ``<pr|qs>`` for the stored ``(pq|rs)``.
        tol: Skip entries below this magnitude.
        float_format: ``printf`` format for the integral value.

    Returns:
        The file body as a string.
    """
    if isinstance(spec, MolecularData):
        h1, h2, ecore = spec.h1, spec.h2, spec.nuclear_repulsion
        n_elec = spec.n_electrons if n_elec is None else n_elec
        spin = spec.spin if spin is None else spin
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

    lines = [
        f" &FCI NORB={norb:4d},NELEC={int(n_elec):2d},MS2={int(spin)},",
        "  ORBSYM=" + ",".join(str(int(x)) for x in (orbsym or [1] * norb)) + ",",
        f"  ISYM={int(isym)},",
        " &END",
    ]

    if h2 is not None:
        h2 = np.asarray(h2, dtype=float)
        pair = [(i, j) for i in range(norb) for j in range(i + 1)]
        for a, (i, j) in enumerate(pair):
            for b in range(a + 1):
                k, l = pair[b]
                value = h2[i, j, k, l]
                if abs(value) <= tol:
                    continue
                if notation == "physicist":
                    # store (ij|kl) under the labels of <ik|jl>
                    lab = (i + 1, k + 1, j + 1, l + 1)
                elif notation == "chemist":
                    lab = (i + 1, j + 1, k + 1, l + 1)
                else:
                    raise ValueError(
                        f"notation must be 'physicist' or 'chemist', got {notation!r}"
                    )
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
