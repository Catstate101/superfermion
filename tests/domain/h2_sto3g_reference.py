"""The tabulated H2/STO-3G record, held to a first-principles recomputation.

Audit row CH-12 compared the record against two published numbers and could not
decide between them: PySCF is absent from this environment, and every other
chemistry test consumes the record's own integrals, so the record could only
ever agree with itself.  A record that only agrees with itself is unverified.

The STO-3G basis for hydrogen is three s-type Gaussians, so every integral is
elementary: overlap and kinetic energy are closed-form, nuclear attraction and
electron repulsion need the Boys function ``F0``, and for two identical centres
the symmetry-adapted MOs are exact.  This module recomputes the integrals from
the basis set definition alone -- no PySCF, no SuperFermion call -- and the test
module alongside it checks the record elementwise against the numbers below.

The 2x2 CI is the exact FCI in this basis for the symmetric ground state:
``|sigma_g^2>`` and ``|sigma_u^2>`` are both gerade while the singly excited
determinants are ungerade, so the ground state does not mix with them.  Its
off-diagonal element is the *exchange* integral ``(01|01)`` -- the
antisymmetrised matrix element of a double excitation -- not the Coulomb
``(00|11)``; using the Coulomb element gives -1.3579875 Ha instead of -1.1373060
and is caught by ``test_record_fci_energy_follows_from_its_own_integrals``.

Nothing here imports pytest and nothing here is collected as a test: the audit
harness ``_xq4_chem.py`` loads this file by path to report the same numbers the
test suite asserts, rather than a copy of them.

The capital single letters are deliberate: ``A``, ``B``, ``C``, ``D`` are nuclear
centres and ``P``, ``Q`` are Gaussian product centres, while ``a``, ``b``, ``c``,
``d`` are the exponents -- lowercasing the centres, as N803/N806 would, collides
with the exponents of the same integral.
"""

import math

import numpy as np

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903  # CODATA 2018
RECORD_D = 0.735  # angstrom, the geometry the record was computed at

# STO-3G for hydrogen: Pople exponents and contraction coefficients
# (Szabo & Ostlund, Appendix B).
ALPHA = (3.42525091, 0.62391373, 0.16885540)
DCOEF = (0.15432897, 0.53532814, 0.44463454)


def boys0(x: float) -> float:
    """``F0(x) = int_0^1 exp(-x t^2) dt``, series at small ``x``."""
    if x < 1e-10:
        return 1.0 - x / 3.0 + x * x / 10.0
    return 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))


def _norm(a: float) -> float:
    return (2.0 * a / np.pi) ** 0.75


def _sq(u, v) -> float:
    return float(np.sum((u - v) ** 2))


def _overlap(a, A, b, B):
    return (np.pi / (a + b)) ** 1.5 * np.exp(-a * b / (a + b) * _sq(A, B))


def _kinetic(a, A, b, B):
    mu = a * b / (a + b)
    return mu * (3.0 - 2.0 * mu * _sq(A, B)) * _overlap(a, A, b, B)


def _nuclear(a, A, b, B, C, z):
    p = a + b
    P = (a * A + b * B) / p
    return (-2.0 * np.pi / p * z * np.exp(-a * b / p * _sq(A, B))
            * boys0(p * _sq(P, C)))


def _eri(a, A, b, B, c, C, d, D):
    p, q = a + b, c + d
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    return (2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
            * np.exp(-a * b / p * _sq(A, B))
            * np.exp(-c * d / q * _sq(C, D))
            * boys0(p * q / (p + q) * _sq(P, Q)))


def _contract2(fn, pair, *args) -> float:
    tot = 0.0
    for a, ca in zip(ALPHA, DCOEF):
        for b, cb in zip(ALPHA, DCOEF):
            tot += ca * cb * _norm(a) * _norm(b) * fn(a, pair[0], b, pair[1],
                                                      *args)
    return tot


def _contract4(centres) -> float:
    A, B, C, D = centres
    tot = 0.0
    for a, ca in zip(ALPHA, DCOEF):
        for b, cb in zip(ALPHA, DCOEF):
            for c, cc in zip(ALPHA, DCOEF):
                for d, cd in zip(ALPHA, DCOEF):
                    tot += (ca * cb * cc * cd * _norm(a) * _norm(b) * _norm(c)
                            * _norm(d) * _eri(a, A, b, B, c, C, d, D))
    return tot


def _contraction_norm(centre) -> float:
    s = sum(DCOEF[i] * DCOEF[j] * _norm(ALPHA[i]) * _norm(ALPHA[j])
            * _overlap(ALPHA[i], centre, ALPHA[j], centre)
            for i in range(3) for j in range(3))
    return 1.0 / math.sqrt(s)


def h2_integrals(d_angstrom: float) -> dict:
    """MO-basis integrals for H2 at ``d``, occupied orbital first."""
    A = np.zeros(3)
    B = np.array([0.0, 0.0, d_angstrom * BOHR_PER_ANGSTROM])
    atoms = (A, B)
    norms = [(_contraction_norm(c)) for c in atoms]

    s = np.empty((2, 2))
    t = np.empty((2, 2))
    v = np.empty((2, 2))
    for i, pi in enumerate(atoms):
        for j, pj in enumerate(atoms):
            nij = norms[i] * norms[j]
            s[i, j] = _contract2(_overlap, (pi, pj)) * nij
            t[i, j] = _contract2(_kinetic, (pi, pj)) * nij
            v[i, j] = sum(_contract2(_nuclear, (pi, pj), c, 1.0) * nij
                          for c in atoms)

    eri = np.empty((2, 2, 2, 2))
    for i, pi in enumerate(atoms):
        for j, pj in enumerate(atoms):
            for k, pk in enumerate(atoms):
                for m, pm in enumerate(atoms):
                    eri[i, j, k, m] = (_contract4((pi, pj, pk, pm))
                                       * norms[i] * norms[j] * norms[k] * norms[m])

    c1 = 1.0 / math.sqrt(2.0 + 2.0 * s[0, 1])
    c2 = 1.0 / math.sqrt(2.0 - 2.0 * s[0, 1])
    C = np.array([[c1, c2], [c1, -c2]])
    h1 = C.T @ (t + v) @ C
    h2 = np.einsum("pi,qj,pqrs,rk,sl->ijkl", C, C, eri, C, C, optimize=True)
    return {"h1": h1, "h2": h2, "s": s, "C": C}


def hf_from(h1, h2, nuc: float) -> float:
    """Closed-shell RHF total energy in a basis where MO 0 is the occupied one."""
    return 2.0 * h1[0, 0] + h2[0, 0, 0, 0] + nuc


def fci_from(h1, h2, nuc: float) -> float:
    """Exact FCI in the gerade ``{|sigma_g^2>, |sigma_u^2>}`` space."""
    coupling = h2[0, 1, 0, 1]
    mat = np.array([[2.0 * h1[0, 0] + h2[0, 0, 0, 0], coupling],
                    [coupling, 2.0 * h1[1, 1] + h2[1, 1, 1, 1]]])
    return float(np.linalg.eigvalsh(mat)[0]) + nuc


def energies(d_angstrom: float) -> tuple:
    got = h2_integrals(d_angstrom)
    nuc = 1.0 / (d_angstrom * BOHR_PER_ANGSTROM)
    return hf_from(got["h1"], got["h2"], nuc), fci_from(got["h1"], got["h2"], nuc)


def golden_min(f, lo: float, hi: float, n: int = 30) -> tuple:
    """Golden-section minimum; returns ``(x, f(x))``."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - invphi * (hi - lo), lo + invphi * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(n):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = f(d)
    return (lo + hi) / 2.0, min(fc, fd)


def bisect(f, lo: float, hi: float, n: int = 40) -> float:
    flo, fhi = f(lo), f(hi)
    assert flo * fhi < 0.0, f"no sign change on [{lo}, {hi}]: {flo}, {fhi}"
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0:
            return mid
        if (fmid < 0.0) == (flo < 0.0):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return 0.5 * (lo + hi)
