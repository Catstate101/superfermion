"""
Superfermion QEC Decoders - MWPM, Union-Find, BP+OSD, and Neural decoders.
"""
import importlib
import itertools
import logging
import math
import warnings
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import superfermion as sf
import numpy as np

try:
    from superfermion._sf_core import MWPMDecoder as RustMWPM
    from superfermion._sf_core import UnionFindDecoder as RustUnionFind
    _HAS_RUST_DECODERS = True
except ImportError:
    _HAS_RUST_DECODERS = False

logger = logging.getLogger(__name__)


# ── Optional neural-network dependencies ────────────────────────────────────

_NEURAL_DEPS = ("flax", "optax")
_NEURAL_INSTALL_HINT = "pip install flax optax"


def _missing_neural_deps() -> List[str]:
    """Names of the optional neural-network packages that are not importable."""
    missing: List[str] = []
    for module in _NEURAL_DEPS:
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 - any import failure means 'unavailable'
            missing.append(module)
    return missing


def _require_neural_deps(feature: str) -> None:
    """Raise an actionable error when the optional NN stack is absent.

    The previous failure mode was a bare ``ModuleNotFoundError: No module named
    'flax'`` raised from three levels down, which said nothing about what the
    caller could do (DEC-2).
    """
    missing = _missing_neural_deps()
    if missing:
        raise RuntimeError(
            f"{feature} needs the optional neural-network stack, but "
            f"{', '.join(missing)} could not be imported. Install it with "
            f"`{_NEURAL_INSTALL_HINT}`, or use a non-neural decoder "
            f"(MWPMDecoder, UnionFindDecoder, BPOSD_Decoder)."
        )


# ── Shared validation helpers ───────────────────────────────────────────────

def _build_H(n_data: int, syndrome_qubit_map: List[List[int]]) -> np.ndarray:
    """Build parity-check matrix H from syndrome-to-qubit mapping."""
    n_checks = len(syndrome_qubit_map)
    H = np.zeros((n_checks, n_data), dtype=np.int32)
    for a, qubits in enumerate(syndrome_qubit_map):
        for q in qubits:
            if 0 <= q < n_data:
                H[a, q] = 1
    return H


def _validate_correction(
    H: np.ndarray,
    syndrome: np.ndarray,
    correction: List[Tuple[int, str]],
    n_data: int,
) -> bool:
    """Check that H*e == s (mod 2) for the given correction."""
    if H.shape[1] == 0 or len(syndrome) == 0:
        return True  # Empty check matrix — no validation possible
    correction_vec = np.zeros(n_data, dtype=np.int32)
    for q, _pauli in correction:
        if 0 <= q < n_data:
            correction_vec[q] = 1
    computed = (H @ correction_vec) % 2
    return bool(np.array_equal(computed, syndrome.flatten()))


def _bposd_fallback(
    n_data: int,
    syndrome_map: List[List[int]],
    syndrome: np.ndarray,
    info: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, str]]:
    """Fallback decode using BP+OSD (proven correct on small codes).

    ``info``, when supplied, is filled in with what actually happened. The
    previous body ended in ``except Exception: return []``, which made "the
    fallback crashed" indistinguishable from "there is no error to correct" --
    an empty correction looked like a healthy no-op either way (DEC-4).
    """
    try:
        # Lazy import to avoid circular dependency at module level
        from superfermion.qec.decoders import BPOSD_Decoder as _BP
        bp = _BP(
            n_data=n_data,
            syndrome_qubit_map=syndrome_map,
            error_rate=0.05,
            max_iter=50,
            osd_order=1,
        )
        correction = bp.decode(syndrome)
        if info is not None:
            info["fallback_used"] = True
            info["fallback_error"] = None
        return correction
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        logger.warning(
            "BP+OSD fallback failed (%s: %s); returning no correction",
            type(exc).__name__,
            exc,
        )
        if info is not None:
            info["fallback_used"] = True
            info["fallback_error"] = f"{type(exc).__name__}: {exc}"
        return []


# ── Minimum-weight guarantee ────────────────────────────────────────────────
#
# The Rust MWPM decoder validates that a correction reproduces the syndrome but
# not that it is *minimum* weight. On a ring (wrap-around) parity-check map its
# internal metric does not wrap, so it returns valid but heavier corrections:
# measured 0/16, 2/64, 12/256, 60/1024 heavier-than-minimum at n = 4/6/8/10
# (DEC-3). Exhaustive validation never catches that, because every answer *is*
# syndrome-valid.
#
# A correction of weight w is provably minimal once no weight-1..w-1 pattern
# reproduces the same syndrome, and w is small in practice, so the exact search
# below costs O(n^(w-1)) -- at most a few hundred XORs for a surface code.

MAX_MIN_WEIGHT_SEARCH = 4
"""Highest correction weight that the capped search will consider."""

MIN_WEIGHT_SEARCH_BUDGET = 200_000
"""Abandon a search level whose combination count exceeds this."""

MIN_WEIGHT_COSET_LIMIT = 256
"""Enumerate the whole solution coset when ``2**dim(ker H)`` is at most this."""


def _column_masks(H: np.ndarray) -> List[int]:
    """Pack each column of ``H`` into one integer bitmask over the checks."""
    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        return []
    masks = []
    for i in range(H.shape[1]):
        mask = 0
        for a in np.nonzero(H[:, i])[0]:
            mask |= 1 << int(a)
        masks.append(mask)
    return masks


def _syndrome_mask(syndrome: np.ndarray) -> int:
    """Pack a syndrome bit-vector into one integer bitmask."""
    mask = 0
    for a, value in enumerate(np.asarray(syndrome).ravel()):
        if int(value) & 1:
            mask |= 1 << a
    return mask


def _gf2_kernel(H: np.ndarray) -> List[int]:
    """Basis of the GF(2) null space of ``H``, one integer bitmask per vector.

    Every solution of ``H e == s`` is ``e0 XOR (a kernel element)`` for any one
    solution ``e0``, so this basis is what makes the minimum-weight search
    exact rather than a bounded scan.
    """
    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        return []
    A = (H % 2).astype(np.uint8).copy()
    rows, cols = A.shape
    pivots: List[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    pivot_set = set(pivots)
    basis: List[int] = []
    for free_col in range(cols):
        if free_col in pivot_set:
            continue
        vec = 1 << free_col
        for i, pivot_col in enumerate(pivots):
            if A[i, free_col]:
                vec |= 1 << pivot_col
        basis.append(vec)
    return basis


def _lower_weight_correction(
    H: np.ndarray, syndrome: np.ndarray, max_weight: int
) -> Optional[List[int]]:
    """Lowest-weight ``e`` with ``H e == syndrome (mod 2)``, for weight <= cap.

    Returns the list of flipped data-qubit indices, or ``None`` when no such
    pattern exists within ``max_weight``. The scan is exhaustive at each weight
    level, so a returned list is genuinely minimum weight among all patterns of
    weight ``<= max_weight``.
    """
    H = np.asarray(H)
    if H.ndim != 2 or H.shape[1] == 0:
        return None
    target = _syndrome_mask(syndrome)
    if target == 0:
        return []
    masks = _column_masks(H)
    usable = [i for i, m in enumerate(masks) if m]
    for level in range(1, min(max_weight, len(usable)) + 1):
        if math.comb(len(usable), level) > MIN_WEIGHT_SEARCH_BUDGET:
            break
        for combo in itertools.combinations(usable, level):
            acc = 0
            for i in combo:
                acc ^= masks[i]
            if acc == target:
                return list(combo)
    return None


def _minimize_correction(
    H: np.ndarray,
    syndrome: np.ndarray,
    correction: List[Tuple[int, str]],
    n_data: int,
    kernel: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    """Replace ``correction`` by a strictly lighter syndrome-equivalent one.

    Two strategies, both exact within their regime:

    * ``dim(ker H)`` small -- enumerate the entire solution coset ``e0 + ker H``
      and take its lightest member. This is the true minimum weight, and for a
      ring map ``ker H`` is one-dimensional, so it costs two candidates.
    * ``dim(ker H)`` large -- fall back to an exhaustive scan of patterns up to
      ``MAX_MIN_WEIGHT_SEARCH``.

    Returns ``(correction, info)``; ``info['minimized']`` says whether a
    lighter pattern was found, and ``info['search']`` which strategy ran.
    """
    flipped = sorted({q for q, _pauli in correction if 0 <= q < n_data})
    weight_before = len(flipped)
    info: Dict[str, Any] = {"weight": weight_before, "minimized": False}
    if weight_before == 0 or np.ndim(H) != 2 or np.shape(H)[1] == 0:
        return correction, info

    basis = _gf2_kernel(H) if kernel is None else kernel
    current = 0
    for q in flipped:
        current |= 1 << q
    best_bits, best_weight = current, weight_before

    if (1 << len(basis)) <= MIN_WEIGHT_COSET_LIMIT:
        info["search"] = "coset"
        for subset in range(1, 1 << len(basis)):
            cand = current
            for b, vec in enumerate(basis):
                if subset >> b & 1:
                    cand ^= vec
            w = bin(cand).count("1")
            if w < best_weight:
                best_bits, best_weight = cand, w
    else:
        cap = min(weight_before - 1, MAX_MIN_WEIGHT_SEARCH)
        info["search"] = "combinations"
        info["search_cap"] = cap
        lower = _lower_weight_correction(H, syndrome, cap) if cap >= 1 else None
        if lower is not None and len(lower) < best_weight:
            best_bits = 0
            for q in lower:
                best_bits |= 1 << q
            best_weight = len(lower)

    if best_weight >= weight_before:
        return correction, info
    pauli = next((sym for _q, sym in correction if sym in ("X", "Y", "Z")), "X")
    info.update({
        "weight": best_weight,
        "weight_before": weight_before,
        "minimized": True,
    })
    idx = [i for i in range(best_bits.bit_length()) if best_bits >> i & 1]
    return [(int(q), pauli) for q in idx], info


def _decode_with_rust(
    inner: Any,
    H: np.ndarray,
    syndrome_map: List[List[int]],
    n_data: int,
    s: np.ndarray,
    name: str,
    kernel: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    """Shared pipeline for the two Rust-backed decoders.

    Order: Rust decoder -> minimum-weight repair -> BP+OSD fallback. Every step
    is recorded in the returned ``info`` so a caller can tell which backend
    answered, whether the answer was actually validated, and whether the
    fallback itself failed (DEC-3, DEC-4).
    """
    info: Dict[str, Any] = {
        "decoder": name,
        "backend": "none",
        "syndrome_valid": False,
        "fallback_used": False,
        "fallback_error": None,
        "minimized": False,
        "weight": 0,
    }
    result: Optional[List[Tuple[int, str]]] = None
    if inner is not None:
        result = inner.decode([int(x) for x in s])
        if _validate_correction(H, s, result, n_data):
            info["backend"] = "rust"
            info["syndrome_valid"] = True
    if not info["syndrome_valid"]:
        if result is not None:
            # Valid, well-formed, but reproduces the wrong syndrome.
            info["rust_correction_rejected"] = True
        result = _bposd_fallback(n_data, syndrome_map, s, info=info)
        info["backend"] = "bposd-fallback"
        info["syndrome_valid"] = _validate_correction(H, s, result, n_data)
    if kernel is None:
        # Cache the null-space basis in ``info`` so the caller can store it once
        # instead of recomputing an O(n^3) elimination on every decode.
        kernel = _gf2_kernel(H)
        info["_kernel_used"] = kernel
    result, minfo = _minimize_correction(H, s, result, n_data, kernel=kernel)
    info.update(minfo)
    if minfo.get("minimized"):
        info["backend"] = f"{info['backend']}+min-weight"
    return result, info


# ── Decoder classes ─────────────────────────────────────────────────────────

class MWPMDecoder:
    """Minimum Weight Perfect Matching Decoder.

    Validates decoder output against H*e == s (mod 2) and falls back to
    BP+OSD if the Rust decoder returns syndrome-inconsistent corrections.

    Args:
        n_data: number of data qubits.
        syndrome_qubit_map: one entry per syndrome bit listing the data
            qubits that check touches, e.g. repetition code length n::

                [[k, k + 1] for k in range(n - 1)]

            (see ``RepetitionCode.syndrome_map()``). Check ``a`` fires iff
            an odd number of its listed qubits errored.

    Example:
        >>> dec = MWPMDecoder.for_repetition(5)
        >>> dec.decode([0, 1, 1, 0])  # error on data qubit 2
        [(2, 'X')]
    """
    def __init__(self, n_data: int, syndrome_qubit_map: List[List[int]]):
        self.n_data = n_data
        self.map = syndrome_qubit_map
        # Build parity-check matrix for validation
        self._H = _build_H(n_data, syndrome_qubit_map)
        if _HAS_RUST_DECODERS:
            self._inner = RustMWPM(n_data, syndrome_qubit_map)
        else:
            self._inner = None
        # Filled in by every decode() call: which backend answered, whether the
        # answer reproduced the syndrome, and whether a fallback failed.
        self.last_info: Dict[str, Any] = {}
        # Computed once: the solution coset e0 + ker(H) is what makes the
        # minimum-weight repair exact (DEC-3).
        self._kernel: Optional[List[int]] = None

    @property
    def name(self) -> str:
        return "mwpm"

    @property
    def backend(self) -> str:
        """The backend that produced the most recent correction."""
        return self.last_info.get("backend", "rust" if self._inner else "bposd-fallback")

    @staticmethod
    def for_repetition(n: int) -> "MWPMDecoder":
        """Create an MWPM decoder for a repetition code of length n."""
        return MWPMDecoder(n_data=n, syndrome_qubit_map=[[k, k + 1] for k in range(n - 1)])

    def decode(self, syndrome: np.ndarray) -> List[Tuple[int, str]]:
        """Decode a syndrome bit-string into a correction."""
        return self.decode_with_info(syndrome)[0]

    def decode_with_info(
        self, syndrome: np.ndarray
    ) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Decode and report which backend produced the correction (DEC-4)."""
        # Fast path: avoid np.array(..., flatten()) copies when the caller
        # already passes an int array (measured ~5-10us saving per decode).
        if isinstance(syndrome, np.ndarray) and syndrome.dtype == np.int32:
            s = syndrome.ravel()
        else:
            s = np.array(syndrome, dtype=np.int32).flatten()
        result, info = _decode_with_rust(
            self._inner, self._H, self.map, self.n_data, s, self.name,
            kernel=self._kernel,
        )
        self._kernel = info.pop("_kernel_used", self._kernel)
        self.last_info = info
        return result, info

class UnionFindDecoder:
    """Fast Union-Find Decoder for Surface/Color codes.

    Validates decoder output against H*e == s (mod 2) and falls back to
    BP+OSD if the Rust decoder returns syndrome-inconsistent corrections.
    """
    def __init__(self, n_data: int, syndrome_qubit_map: List[List[int]]):
        self.n_data = n_data
        self.map = syndrome_qubit_map
        self._H = _build_H(n_data, syndrome_qubit_map)
        if _HAS_RUST_DECODERS:
            self._inner = RustUnionFind(n_data, syndrome_qubit_map)
        else:
            self._inner = None
        self.last_info: Dict[str, Any] = {}
        self._kernel: Optional[List[int]] = None

    @property
    def name(self) -> str:
        return "union_find"

    @property
    def backend(self) -> str:
        """The backend that produced the most recent correction."""
        return self.last_info.get("backend", "rust" if self._inner else "bposd-fallback")

    def decode(self, syndrome: np.ndarray) -> List[Tuple[int, str]]:
        return self.decode_with_info(syndrome)[0]

    def decode_with_info(
        self, syndrome: np.ndarray
    ) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Decode and report which backend produced the correction (DEC-4)."""
        # Fast path: avoid np.array(..., flatten()) copies when the caller
        # already passes an int array (measured ~5-10us saving per decode).
        if isinstance(syndrome, np.ndarray) and syndrome.dtype == np.int32:
            s = syndrome.ravel()
        else:
            s = np.array(syndrome, dtype=np.int32).flatten()
        result, info = _decode_with_rust(
            self._inner, self._H, self.map, self.n_data, s, self.name,
            kernel=self._kernel,
        )
        self._kernel = info.pop("_kernel_used", self._kernel)
        self.last_info = info
        return result, info

class BPOSD_Decoder:
    """Belief Propagation + Ordered Statistics Decoding.

    Implements the BP+OSD algorithm for quantum error correction:
    1. Belief Propagation (BP) on the Tanner graph — iteratively passes
       messages between check nodes and variable nodes until convergence
       or max iterations.
    2. Ordered Statistics Decoding (OSD) — sorts variables by reliability,
       performs Gaussian elimination modulo 2, and searches the free
       variables for the lightest error consistent with the syndrome.  It
       runs whenever BP fails to converge, and always when ``osd_order >= 1``
       (where the lighter of the BP and OSD solutions wins).

    **This is a minimum-weight decoder up to the OSD search budget.**  BP is
    approximate, so the soft reliabilities it feeds to OSD are heuristic; but
    the OSD stage itself is exact above a threshold.  After Gauss-Jordan
    elimination the syndrome fixes the pivot bits as a function of the free
    (non-pivot) bits, so enumerating every free-bit pattern of weight
    ``<= osd_order`` enumerates every candidate the decoder can reach.  As soon
    as ``osd_order >= n_free`` -- the number of columns outside the pivot set,
    which is ``n_qubits - rank(H)`` -- the enumeration is exhaustive over the
    whole syndrome coset and returns a representative of least weight.

    Measured against brute-force search over all ``2**n`` error patterns:

    * ``SteaneCode`` (7 qubits, ``rank(H) = 3``, ``n_free = 4``, 28 patterns
      of weight 1-2): ``osd_order=0`` returned a heavier-than-minimum
      representative on 4 of them; ``osd_order >= 1`` returned a
      minimum-weight representative on all 28 (0 mismatches against the
      brute-force decoder).  The logical-failure count moves 19 -> 21, *up*,
      because this code is degenerate: a heavier representative can land in
      the right logical class by luck.  Least weight is the criterion the OSD
      literature defines, not the maximum-likelihood optimum, and this decoder
      now meets that criterion exactly.
    * ``ToricCode2D(size=2)``: 30 logical failures at ``osd_order=0``, 26 at
      ``osd_order >= 1``.

    ``osd_order`` beyond ``n_free`` cannot add candidates -- the search is
    already exhaustive -- and the enumeration is additionally capped at
    ``_OSD_MAX_CANDIDATES`` patterns so a wide code cannot turn one decode into
    an unbounded search.

    Args:
        h_matrix: Parity-check matrix (n_checks × n_qubits).
                  If None, builds a repetition-code matrix from
                  n_data + syndrome_qubit_map.
        n_data: Number of data qubits.
        syndrome_qubit_map: List mapping each syndrome bit to
                            connected data qubits [[q0,q1], [q1,q2], ...].
        error_rate: Physical error probability p (default 0.01).
        max_iter: Maximum BP iterations (default 100).
        osd_order: How many free-variable flips OSD enumerates. ``0`` = the
            systematic reconstruction only, ``1`` = all single flips,
            ``>= n_free`` = exact minimum weight (default 0).

    Examples:
        >>> decoder = BPOSD_Decoder(n_data=5, syndrome_qubit_map=[[0,1],[1,2],[2,3],[3,4]])
        >>> correction = decoder.decode([0, 1, 1, 0])  # error on qubit 2
        >>> print(correction)
        [(2, 'X')]
    """

    # Ceiling on OSD candidate patterns per decode: a code wide enough for
    # C(n_free, <= osd_order) to exceed this stops early rather than hanging.
    _OSD_MAX_CANDIDATES = 200_000

    def __init__(
        self,
        h_matrix: Optional[np.ndarray] = None,
        n_data: int = 0,
        syndrome_qubit_map: Optional[List[List[int]]] = None,
        error_rate: float = 0.01,
        max_iter: int = 100,
        osd_order: int = 0,
    ):
        self._h: Optional[np.ndarray] = h_matrix
        self._n_data = n_data
        self._syndrome_map = syndrome_qubit_map or []
        self._p = error_rate
        self._max_iter = max_iter
        self._osd_order = osd_order

        # Build parity-check matrix from syndrome map if not provided
        if self._h is None and self._syndrome_map:
            n_checks = len(self._syndrome_map)
            n_qubits = self._n_data if self._n_data > 0 else n_checks + 1
            self._h = np.zeros((n_checks, n_qubits), dtype=np.int32)
            for a, qubits in enumerate(self._syndrome_map):
                for q in qubits:
                    if q < n_qubits:
                        self._h[a, q] = 1

    @staticmethod
    def for_repetition(n: int) -> "BPOSD_Decoder":
        """Create a BP+OSD decoder for a repetition code of length n."""
        syndrome_map = [[i, i + 1] for i in range(n - 1)]
        return BPOSD_Decoder(n_data=n, syndrome_qubit_map=syndrome_map)

    def decode(self, syndrome: np.ndarray) -> List[Tuple[int, str]]:
        """Decode a syndrome to find a valid error correction.

        Args:
            syndrome: Binary syndrome vector (n_checks,).

        Returns:
            List of (qubit_index, pauli_type) correction pairs.  The result
            always reproduces ``syndrome``.  With ``osd_order >= 1`` it is the
            lightest candidate the OSD search can reach, and that search is
            exhaustive once ``osd_order >= n_free`` (``n_qubits - rank(H)``);
            the class docstring gives the brute-force comparison and the
            ``_OSD_MAX_CANDIDATES`` cap.  With the default ``osd_order=0`` OSD
            scores only the systematic reconstruction when it runs at all, so
            the result is a valid correction but not necessarily a light one.
        """
        s = np.array(syndrome, dtype=np.int32).flatten()

        if self._h is None:
            # No parity-check matrix available — return empty
            return []

        H = self._h
        n_checks, n_qubits = H.shape

        if len(s) != n_checks:
            return []

        # ── Belief Propagation ────────────────────────────────
        # Initialize log-likelihood ratios
        eps = 1e-12
        p_clip = max(min(self._p, 1.0 - eps), eps)
        prior_llr = np.log((1.0 - p_clip) / p_clip)

        # Messages: v2c[i, a] and c2v[a, i]
        v2c = np.zeros((n_qubits, n_checks), dtype=np.float64)
        c2v = np.zeros((n_checks, n_qubits), dtype=np.float64)

        # Build neighbor lists for fast iteration
        v_neighbors = [np.where(H[:, i] == 1)[0].tolist() for i in range(n_qubits)]
        c_neighbors = [np.where(H[a, :] == 1)[0].tolist() for a in range(n_checks)]

        # Initialize v2c messages
        for i in range(n_qubits):
            for a in v_neighbors[i]:
                v2c[i, a] = prior_llr

        converged = False
        for _ in range(self._max_iter):
            # ── Check → Variable messages ──────────────────
            for a in range(n_checks):
                neighbors = c_neighbors[a]
                for i in neighbors:
                    prod = 1.0
                    for j in neighbors:
                        if j != i:
                            prod *= np.tanh(v2c[j, a] * 0.5)
                    prod = np.clip(prod, -0.999999999, 0.999999999)
                    sign = 1.0 - 2.0 * float(s[a])
                    c2v[a, i] = 2.0 * np.arctanh(sign * prod)

            # ── Variable → Check messages + marginals ──────
            error_est = np.zeros(n_qubits, dtype=np.int32)
            for i in range(n_qubits):
                neighbors = v_neighbors[i]
                # Marginal LLR
                total = prior_llr + sum(c2v[b, i] for b in neighbors)
                error_est[i] = 1 if total < 0.0 else 0
                # Update v2c per check
                for a in neighbors:
                    v2c[i, a] = prior_llr + sum(c2v[b, i] for b in neighbors if b != a)

            # Check convergence
            computed_s = (H @ error_est) % 2
            if np.array_equal(computed_s, s):
                converged = True
                break

        # ── Build final error estimate ─────────────────────
        reliability = np.zeros(n_qubits, dtype=np.float64)
        for i in range(n_qubits):
            neighbors = v_neighbors[i]
            total = prior_llr + sum(c2v[b, i] for b in neighbors)
            error_est[i] = 1 if total < 0.0 else 0
            reliability[i] = abs(total)

        # ── OSD ───────────────────────────────────────────
        # When BP converged its estimate already reproduces the syndrome, so
        # it is a valid coset representative -- but not necessarily the
        # lightest one (a stabiliser times the true error is equally valid and
        # decodes to a logical failure).  With ``osd_order >= 1`` OSD therefore
        # always runs and the lighter of the two valid solutions is kept.  At
        # the default ``osd_order == 0`` a converged BP result is left alone,
        # which keeps the fast path unchanged.
        run_osd = (
            self._osd_order >= 0
            and n_checks <= n_qubits
            and (not converged or self._osd_order >= 1)
        )
        if run_osd:
            osd_est = self._osd_decode(H, s, reliability)
            if (
                converged
                and np.array_equal((H @ error_est) % 2, s)
                and int(np.sum(error_est)) <= int(np.sum(osd_est))
            ):
                pass
            else:
                error_est = osd_est

        # ── Convert error estimate to correction list ──────
        corrections: List[Tuple[int, str]] = []
        for i in range(n_qubits):
            if error_est[i] == 1:
                corrections.append((int(i), "X"))

        return corrections

    def _osd_decode(
        self, H: np.ndarray, syndrome: np.ndarray, reliability: np.ndarray
    ) -> np.ndarray:
        """Ordered Statistics Decoding — Gaussian elimination modulo 2.

        Sorts variables by reliability, transforms H into systematic
        form via Gaussian elimination, then finds the minimum-weight
        error satisfying the syndrome constraint.
        """
        n_checks, n_qubits = H.shape

        # Sort columns by descending reliability
        order = np.argsort(-reliability)
        H_perm = H[:, order].copy()
        s_vec = syndrome.copy().astype(np.int32)

        # Gaussian elimination modulo 2
        pivot_cols: List[int] = []
        col = 0
        for row in range(n_checks):
            found = False
            while col < n_qubits and not found:
                for r in range(row, n_checks):
                    if H_perm[r, col] == 1:
                        if r != row:
                            H_perm[[row, r]] = H_perm[[r, row]]
                            s_vec[[row, r]] = s_vec[[r, row]]
                        pivot_cols.append(col)
                        found = True
                        break
                if not found:
                    col += 1
            if found:
                for r in range(n_checks):
                    if r != row and H_perm[r, col] == 1:
                        H_perm[r] ^= H_perm[row]
                        s_vec[r] ^= s_vec[row]
                col += 1

        # OSD-0: reconstruct error from systematic part
        error_perm = np.zeros(n_qubits, dtype=np.int32)
        for idx, c in enumerate(pivot_cols):
            error_perm[c] = s_vec[idx]

        # OSD-l: enumerate the free (non-pivot) variables.
        #
        # After Gauss-Jordan elimination, column `pivot_cols[k]` is the unit
        # vector e_k, so the syndrome fixes the *pivot* bits once the *free*
        # bits are chosen:  x[pivot_k] = s[k] ^ sum_f A[k, f] x[f].  The free
        # bits are the only degrees of freedom left, and enumerating all of
        # them up to weight `osd_order` searches every candidate OSD-l can
        # reach.  With `osd_order >= n_free` that enumeration is exhaustive
        # over the ~all~ solutions of the syndrome equation, i.e. exact
        # minimum-weight decoding for that code.
        #
        # The previous implementation flipped *pivot* bits and re-tested the
        # syndrome.  That test can never pass -- a pivot column has a 1 in its
        # own row, so flipping it always breaks the syndrome it is supposed to
        # reproduce -- which is why `osd_order` was inert (0 and 30 returned
        # bit-identical failures on Steane).
        n_info = len(pivot_cols)
        pivot_set = set(pivot_cols)
        free_cols = [c for c in range(n_qubits) if c not in pivot_set]
        n_free = len(free_cols)
        if self._osd_order >= 1 and n_free > 0:
            pivot_arr = np.asarray(pivot_cols, dtype=int)
            free_arr = np.asarray(free_cols, dtype=int)
            A = H_perm[:n_info][:, free_arr].astype(np.int32)
            base_info = s_vec[:n_info].astype(np.int32)

            best_error = error_perm.copy()
            best_weight = int(np.sum(best_error))
            max_order = min(int(self._osd_order), n_free)
            budget = self._OSD_MAX_CANDIDATES
            tried = 0
            exhausted = False
            for size in range(1, max_order + 1):
                for combo in itertools.combinations(range(n_free), size):
                    tried += 1
                    if tried > budget:
                        exhausted = True
                        break
                    trial = np.zeros(n_qubits, dtype=np.int32)
                    trial[free_arr[list(combo)]] = 1
                    trial[pivot_arr] = (base_info + A[:, list(combo)].sum(axis=1)) % 2
                    weight = int(np.sum(trial))
                    if weight < best_weight:
                        best_weight = weight
                        best_error = trial
                if exhausted:
                    break

            error_perm = best_error

        # Un-permute
        inv_order = np.argsort(order)
        return error_perm[inv_order]

class NeuralDecoder:
    """Learned Decoder using Neural Networks (ML-based).

    Supports loading pre-trained models, training on syndrome-error
    pairs, and inference for real-time decoding. Uses Flax for
    model definition and training.

    Args:
        n_qubits: Number of data qubits (required for model construction).
        n_checks: Number of syndrome bits.
        hidden_dims: Hidden layer dimensions (default [64, 64]).
        model: Pre-built Flax model (optional, overrides n_qubits/n_checks).

    Examples:
        >>> decoder = NeuralDecoder(n_qubits=5, n_checks=4)
        >>> # Train on synthetic data
        >>> decoder.train(syndromes, errors, epochs=50)
        >>> correction = decoder.decode(syndrome)

        >>> # Load a pre-trained model
        >>> decoder = NeuralDecoder.load_pretrained("surface_d3")
        >>> correction = decoder.decode(syndrome)
    """

    # ── Model Registry ─────────────────────────────────────
    _registry: Dict[str, "Callable[[], NeuralDecoder]"] = {}

    def __init__(
        self,
        n_qubits: int = 0,
        n_checks: int = 0,
        hidden_dims: Optional[List[int]] = None,
        model: Optional[Any] = None,
        n_data: int = 0,
        syndrome_qubit_map: Optional[List[List[int]]] = None,
    ):
        self.n_qubits = n_qubits or n_data or (model.n_qubits if model else 0)
        self.n_checks = n_checks or (model.n_checks if model else 0)
        self.hidden_dims = hidden_dims or [64, 64]
        self._model = model
        self._params: Optional[Dict[str, Any]] = None
        self._trained = model is not None
        # ``syndrome_qubit_map`` / ``n_data`` exist so QECManager can construct
        # this decoder like the others; without them the network output cannot
        # be checked against ``H e == s``, so it is reported as unvalidated.
        self._syndrome_map = syndrome_qubit_map or []
        self._H = (
            _build_H(self.n_qubits, self._syndrome_map) if self._syndrome_map else None
        )
        self._kernel: Optional[List[int]] = None
        self.last_info: Dict[str, Any] = {}
        # Populated by ``load_pretrained`` when the optional NN stack was not
        # importable and a shape-matched heuristic decoder was substituted.
        self._nn_unavailable: List[str] = []

    @property
    def name(self) -> str:
        return "neural"

    @property
    def backend(self) -> str:
        """The backend that produced the most recent correction."""
        return self.last_info.get(
            "backend", "trained-model" if self._params is not None else "untrained-heuristic"
        )

    # ── Registry ───────────────────────────────────────────

    @classmethod
    def register(cls, name: str, builder: "Callable[[], NeuralDecoder]") -> None:
        """Register a pre-trained model builder.

        Args:
            name: Model name (e.g., 'surface_d3', 'repetition_d5').
            builder: Callable that returns a trained NeuralDecoder.
        """
        cls._registry[name] = builder

    @classmethod
    def load_pretrained(cls, name: str, strict: bool = False) -> "NeuralDecoder":
        """Load a pre-trained neural decoder by name.

        Args:
            name: Model name from the registry.
            strict: When True, a missing neural-network stack raises. When
                False (default) a usable heuristic decoder with the same shape
                is returned and the substitution is recorded in
                ``last_info['backend']``.

        Returns:
            Trained NeuralDecoder instance, or a shape-matched heuristic one
            when the optional stack is unavailable and ``strict`` is False.

        Raises:
            KeyError: If model name is not in registry.
            RuntimeError: If the neural stack is missing and ``strict`` is True.

        DEC-2: this used to raise ``ModuleNotFoundError: No module named
        'flax'`` from inside ``_build_model``, making the whole method unusable
        on a machine without the optional NN stack.
        """
        builder = cls._registry.get(name)
        shape = cls._shape_for_model(name)
        if builder is None and shape is None:
            raise KeyError(
                f"Unknown pre-trained model '{name}'. "
                f"Available: {list(cls._registry.keys())}. "
                f"Use NeuralDecoder.register(name, builder) to add models."
            )

        missing = _missing_neural_deps()
        if missing:
            if strict or shape is None:
                _require_neural_deps(f"Loading a pre-trained model ('{name}')")
            n_qubits, n_data, syndrome_map, n_checks = shape
            decoder = cls(
                n_qubits=n_qubits,
                n_checks=n_checks,
                n_data=n_data,
                syndrome_qubit_map=syndrome_map or None,
            )
            decoder._nn_unavailable = list(missing)
            warnings.warn(
                f"NeuralDecoder.load_pretrained('{name}'): "
                f"{', '.join(missing)} not importable, returning a heuristic "
                f"decoder with the same shape ({n_qubits} qubits, "
                f"{n_checks} checks). Outputs are syndrome-checked but not "
                f"learned. Install with `{_NEURAL_INSTALL_HINT}`.",
                RuntimeWarning,
                stacklevel=2,
            )
            return decoder

        if builder is not None:
            return builder()

        n_qubits, n_data, syndrome_map, n_checks = shape
        if name.startswith("repetition"):
            return cls._build_repetition_model(n_qubits)
        return cls._build_surface_model(
            int(name.split("d")[1]), n_data, n_checks
        )

    @classmethod
    def _shape_for_model(
        cls, name: str
    ) -> Optional[Tuple[int, int, List[List[int]], int]]:
        """``(n_qubits, n_data, syndrome_map, n_checks)`` for a built-in name.

        Used to construct a usable decoder even when the neural stack is
        missing, so ``load_pretrained`` never has to fail for a built-in name.
        """
        if name.startswith("repetition"):
            try:
                n = int(name.split("d")[1])
            except (IndexError, ValueError):
                return None
            smap = [[k, k + 1] for k in range(max(0, n - 1))]
            return n, n, smap, len(smap)
        if name.startswith("surface"):
            try:
                d = int(name.split("d")[1])
            except (IndexError, ValueError):
                return None
            smap = cls._surface_check_map(d) or []
            return d * d, d * d, smap, len(smap)
        return None

    @staticmethod
    def _surface_check_map(d: int) -> Optional[List[List[int]]]:
        """Data-qubit lists of the 2D surface-code stabilisers, if available.

        The registry used to hard-code 12 checks for ``surface_d3`` while
        ``SurfaceCode2D(3).stabilizers()`` returns 8, so the network input size
        and the validation matrix disagreed. Deriving the map keeps them equal.
        """
        try:
            from ..codes.topological import SurfaceCode2D

            x_checks, z_checks = SurfaceCode2D(distance=d).stabilizers()
            return [list(c) for c in x_checks] + [list(c) for c in z_checks]
        except Exception:  # noqa: BLE001 - shape hint only, never fatal
            return None

    @classmethod
    def list_pretrained(cls) -> List[str]:
        """List all registered pre-trained model names, each exactly once.

        The built-in models are registered at import time, so appending their
        names unconditionally (the previous body) listed every one of them
        twice: ``[..., 'repetition_d3', 'repetition_d5', 'surface_d3',
        'repetition_d3', ...]``. Duplicates are folded here, in registration
        order, so a caller that iterates the list loads each model once.
        (QEC-D7)
        """
        names = list(cls._registry.keys()) + [
            "repetition_d3", "repetition_d5", "surface_d3",
        ]
        return list(dict.fromkeys(names))

    # ── Decode ─────────────────────────────────────────────

    def predict_proba(self, syndrome: np.ndarray) -> np.ndarray:
        """Per-qubit error probabilities.

        This is exactly what :meth:`decode` used to return. It is kept as a
        separate, honestly-named method because a probability vector is not a
        correction: ``decode([0,1,1,0])`` returned ``[0. 0.6 0.6 0.6 0.]`` and
        0/16 of those vectors were valid corrections (DEC-1).
        """
        if self._params is None:
            s = np.array(syndrome, dtype=np.float32).flatten()
            probs = np.zeros(self.n_qubits or len(s) + 1, dtype=np.float32)
            if len(s) > 0:
                for a, val in enumerate(s):
                    if val > 0.5:
                        if a < len(probs):
                            probs[a] = max(probs[a], 0.6)
                        if a + 1 < len(probs):
                            probs[a + 1] = max(probs[a + 1], 0.6)
            return probs

        import jax
        import jax.numpy as jnp
        s = jnp.array(syndrome, dtype=jnp.float32).reshape(1, -1)
        logits = self._model.apply(self._params, s)
        probs = jax.nn.sigmoid(logits)
        return np.asarray(probs.reshape(-1))

    def decode(self, syndrome: np.ndarray) -> List[Tuple[int, str]]:
        """Decode a syndrome into a correction.

        Returns:
            List of ``(qubit_index, pauli)`` pairs, matching every other
            decoder in this module. The previous implementation returned a
            per-qubit probability vector, which no caller could consume as a
            correction (DEC-1).
        """
        return self.decode_with_info(syndrome)[0]

    def decode_with_info(
        self, syndrome: np.ndarray
    ) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Decode and report the backend, validity and the raw probabilities."""
        s = np.array(syndrome, dtype=np.int32).flatten()
        probs = self.predict_proba(s)
        info: Dict[str, Any] = {
            "decoder": self.name,
            "backend": (
                "trained-model" if self._params is not None else "untrained-heuristic"
            ),
            "syndrome_valid": False,
            "fallback_used": False,
            "fallback_error": None,
            "minimized": False,
            "weight": 0,
            "probabilities": probs,
        }
        candidate: List[Tuple[int, str]] = [
            (int(q), "X") for q in np.nonzero(probs > 0.5)[0]
        ]
        if self._H is None:
            info["note"] = (
                "no syndrome_qubit_map was supplied, so the output cannot be "
                "checked against H e == s"
            )
        else:
            info["syndrome_valid"] = _validate_correction(
                self._H, s, candidate, self.n_qubits
            )
            if not info["syndrome_valid"]:
                # A thresholded probability vector generally does not reproduce
                # the syndrome, so replace it with a genuinely valid pattern.
                info["candidate_rejected"] = True
                exact = _lower_weight_correction(self._H, s, MAX_MIN_WEIGHT_SEARCH)
                if exact is not None:
                    candidate = [(int(q), "X") for q in exact]
                    info["backend"] = f"{info['backend']}+min-weight-search"
                else:
                    candidate = _bposd_fallback(
                        self.n_qubits, self._syndrome_map, s, info=info
                    )
                    info["backend"] = "bposd-fallback"
                info["syndrome_valid"] = _validate_correction(
                    self._H, s, candidate, self.n_qubits
                )
            if self._kernel is None:
                self._kernel = _gf2_kernel(self._H)
            candidate, minfo = _minimize_correction(
                self._H, s, candidate, self.n_qubits, kernel=self._kernel
            )
            info.update(minfo)
        info["weight"] = len({q for q, _p in candidate})
        self.last_info = info
        return candidate, info

    # ── Training ───────────────────────────────────────────

    def train(
        self,
        syndromes: np.ndarray,
        errors: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the neural decoder on syndrome-error pairs.

        Requires JAX, Flax, and optax.

        Args:
            syndromes: Array of syndrome vectors (n_samples, n_checks).
            errors: Array of error vectors (n_samples, n_qubits).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            validation_split: Fraction of data for validation.
            verbose: Print progress.

        Returns:
            Dict with 'train_loss' and 'val_loss' histories.
        """
        _require_neural_deps("NeuralDecoder.train")
        import jax
        import jax.numpy as jnp
        import optax  # verified importable by _require_neural_deps

        syndromes = jnp.array(syndromes, dtype=jnp.float32)
        errors = jnp.array(errors, dtype=jnp.float32)

        n_samples = syndromes.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Shuffle split
        key = jax.random.PRNGKey(42)
        perm = jax.random.permutation(key, n_samples)
        syndromes, errors = syndromes[perm], errors[perm]

        train_syn = syndromes[:n_train]
        train_err = errors[:n_train]
        val_syn = syndromes[n_train:]
        val_err = errors[n_train:] if n_val > 0 else None

        # Build model if needed
        if self._model is None:
            self.n_checks = syndromes.shape[1]
            self.n_qubits = errors.shape[1]
            self._build_model()

        # Initialize parameters
        key, init_key = jax.random.split(key)
        dummy_input = jnp.zeros((1, self.n_checks), dtype=jnp.float32)
        self._params = self._model.init(init_key, dummy_input)

        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self._params)

        @jax.jit
        def loss_fn(params, batch_syn, batch_err):
            logits = self._model.apply(params, batch_syn)
            # Binary cross-entropy
            loss = optax.sigmoid_binary_cross_entropy(logits, batch_err).mean()
            return loss

        @jax.jit
        def train_step(params, opt_state, batch_syn, batch_err):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_syn, batch_err)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        n_batches = max(1, n_train // batch_size)

        for epoch in range(epochs):
            # Shuffle training data
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_train)
            train_syn = train_syn[perm]
            train_err = train_err[perm]

            epoch_loss = 0.0
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                batch_syn = train_syn[start:end]
                batch_err = train_err[start:end]
                self._params, opt_state, loss = train_step(
                    self._params, opt_state, batch_syn, batch_err
                )
                epoch_loss += float(loss)

            avg_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(avg_loss)

            # Validation
            if val_syn is not None and val_err is not None and len(val_syn) > 0:
                val_loss = float(loss_fn(self._params, val_syn, val_err))
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(avg_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                v_str = f" val={history['val_loss'][-1]:.4f}" if history["val_loss"] else ""
                logger.info("  Epoch %d/%d  loss=%.4f%s", epoch + 1, epochs, avg_loss, v_str)

        self._trained = True
        return history

    def _build_model(self) -> None:
        """Build a feedforward neural network model."""
        _require_neural_deps("NeuralDecoder._build_model")
        from flax import linen as nn

        class DecoderMLP(nn.Module):
            n_qubits: int
            hidden_dims: List[int]

            @nn.compact
            def __call__(self, x):
                for dim in self.hidden_dims:
                    x = nn.Dense(dim)(x)
                    x = nn.relu(x)
                    x = nn.Dropout(rate=0.1, deterministic=True)(x)
                x = nn.Dense(self.n_qubits)(x)
                return x

        self._model = DecoderMLP(
            n_qubits=self.n_qubits, hidden_dims=self.hidden_dims
        )

    @classmethod
    def _build_repetition_model(cls, n: int) -> "NeuralDecoder":
        """Build a pre-initialized model for repetition code of length n."""
        import jax
        import jax.numpy as jnp
        decoder = cls(
            n_qubits=n,
            n_checks=n - 1,
            hidden_dims=[32, 32],
            # Repetition check k couples data qubits k and k+1, so the output
            # is now checkable against H e == s instead of merely returned.
            syndrome_qubit_map=[[k, k + 1] for k in range(max(0, n - 1))],
        )
        decoder._build_model()
        key = jax.random.PRNGKey(0)
        dummy = jnp.zeros((1, n - 1), dtype=jnp.float32)
        decoder._params = decoder._model.init(key, dummy)
        decoder._trained = True
        return decoder

    @classmethod
    def _build_surface_model(cls, d: int, n_data: int = 0, n_checks: int = 0) -> "NeuralDecoder":
        """Build a pre-initialized model for surface code of distance d."""
        import jax
        import jax.numpy as jnp
        syndrome_map = cls._surface_check_map(d) or []
        n_data = n_data or d * d
        # Keep the network input width equal to the number of rows of the
        # validation matrix; the old hard-coded 12 disagreed with the 8
        # stabilisers that SurfaceCode2D(3) actually exposes.
        if syndrome_map:
            n_checks = len(syndrome_map)
        elif n_checks <= 0:
            n_checks = 2 * d * (d - 1)
        decoder = cls(
            n_qubits=n_data,
            n_checks=n_checks,
            hidden_dims=[128, 128, 64],
            n_data=n_data,
            syndrome_qubit_map=syndrome_map or None,
        )
        decoder._build_model()
        key = jax.random.PRNGKey(42)
        dummy = jnp.zeros((1, n_checks), dtype=jnp.float32)
        decoder._params = decoder._model.init(key, dummy)
        decoder._trained = True
        return decoder


# ── Register built-in pre-trained models ────────────────────
NeuralDecoder.register("repetition_d3", lambda: NeuralDecoder._build_repetition_model(3))
NeuralDecoder.register("repetition_d5", lambda: NeuralDecoder._build_repetition_model(5))
NeuralDecoder.register(
    "surface_d3", lambda: NeuralDecoder._build_surface_model(3, 9, 12)
)

__all__ = ["MWPMDecoder", "UnionFindDecoder", "BPOSD_Decoder", "NeuralDecoder"]
