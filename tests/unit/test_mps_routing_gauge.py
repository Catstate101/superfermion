"""The routed (lazy-SWAP) MPS path with the mixed-canonical gauge on.

§39 measured the gauge only on circuits whose two-qubit gates are adjacent on
the chain, which is the case that never reaches `apply_2q_gate_routed` or
`apply_gate_lazy_routed`.  A circuit with any non-adjacent two-qubit gate is
sent by `QuantumDAG::_evolve_into` to `_evolve_into_lazy`, whose routing helper
`apply_lazy_swap_phys` applies each SWAP **through `apply_2q_gate`** -- so every
routing move hits the gauge check and, at a binding cap, pays a real QR/LQ.
§39.8 called that the one shape where the gauge's cost could exceed the +29%
of the adjacency-only ladder.  `_sfpl_bench/_probe_routing_gauge.py` measured it
at n=16, chi=128, 948 gates: the gauge costs 1.05-1.14x on routed circuits
against 1.32x on the adjacent control, and buys the same 8.35x in accuracy.

Two properties from that run are pinned here, because both would regress
silently:

  1. **Wire-map invariance.**  A rewiring `q -> (q*s) % n` of the same circuit
     is the same state with the sites permuted, and the router moves the wires
     rather than truncating them, so the *achieved* 1-fid must not depend on
     `s` at all.  Measured at n=16/chi=128: 9.884e-03 for s = 1, 3, 5 and 7.
     A router that reordered truncations, or a gauge that only fired on
     adjacent gates, would break this.
  2. **The permutation must be readable.**  `State.perm()` is the map onto
     the physical sites the router parked each wire on.  `State.numpy()` on
     an `mps` handle has answered in qubit order since the routed-lane
     readout repair (`WIRE_ORDER_READOUT`); `perm()` is kept so the routing
     stays observable, and `_site_view()` below rebuilds the raw site order
     from it.  That view must be near-orthogonal to the reference -- if it
     is not, `perm()` no longer describes the routing and the honesty check
     on the routed lane is gone.

The fixture is calibrated: `__main__` prints the numbers, and
`test_gauge_still_pays_when_routed` asserts the cap actually binds first, so a
fixture that drifts into the exact regime fails loudly instead of passing.
"""
from __future__ import annotations

import functools
import os

import numpy as np

import superfermion as sf

N = 12
CYCLES = 10
# Calibrated in `__main__` at n=12: 1-fid 4.70e-03 gauged, 2.43e-02 greedy
# (gain 5.17x).  chi=16 truncates so hard that both sides are junk (3.2e-01 vs
# 6.0e-01, gain 1.87x) -- the gauge cannot show its value there.
CHI = 48
SEED = 20261119
STRIDES = (1, 5)          # 1 = adjacency only (control), 5 = routed (gcd(5,12)=1)
# The gauge's measured gain on this fixture; asserted as a lower bound.
MIN_GAIN = 3.0


def _circuit(n, cycles, stride, seed=SEED):
    """`_eq_full.gates_brick` under the wire map q -> (q*stride) % n.

    stride=1 is byte-for-byte the bench's brickwork; any other stride keeps the
    gate count, the RNG draw order and the circuit's unitary action (up to the
    relabeling), and only moves the two-qubit gates apart on the chain.
    """
    if np.gcd(stride, n) != 1:
        raise ValueError(f"stride {stride} is not coprime with n={n}")
    wire = lambda q: (q * stride) % n
    rng = np.random.default_rng(seed)
    c = sf.Circuit(n)
    for _ in range(cycles):
        for q in range(n):
            c = c.ry(float(rng.uniform(-np.pi, np.pi)), wire(q))
            c = c.rz(float(rng.uniform(-np.pi, np.pi)), wire(q))
        for q in range(0, n - 1, 2):
            c = c.cx(wire(q), wire(q + 1))
        for q in range(n):
            c = c.ry(float(rng.uniform(-np.pi, np.pi)), wire(q))
            c = c.rz(float(rng.uniform(-np.pi, np.pi)), wire(q))
        for q in range(1, n - 1, 2):
            c = c.cx(wire(q), wire(q + 1))
    return c


def _infidelity(vec, ref):
    """1 - |<ref|vec>|^2 / (|ref|^2 |vec|^2), so survivor norm is allowed."""
    vec = np.asarray(vec).ravel()
    ov = np.vdot(ref, vec)
    return max(0.0, 1.0 - float(abs(ov) ** 2 / (np.vdot(ref, ref).real
                                               * np.vdot(vec, vec).real)))


def _site_view(wire_vec, perm):
    """Re-label a qubit-order vector into physical-*site* order.

    `perm[wire] = site`, so bit `q` of a qubit-order index names a wire and
    bit `s` of a site-order index names whatever wire the router parked on
    site `s`.  This is the layout the dense amplitudes had before the
    routed-lane readout repair; the test keeps it visible so `perm()` is
    still checkable rather than decorative.
    """
    n = len(perm)
    wire_vec = np.asarray(wire_vec).ravel()
    out = np.empty_like(wire_vec)
    for site_index in range(1 << n):
        wire_index = 0
        for q in range(n):
            if (site_index >> perm[q]) & 1:
                wire_index |= 1 << q
        out[site_index] = wire_vec[wire_index]
    return out


def _run(circuit, chi, gauge):
    """Evolve at `chi` with the gauge forced on/off; return qubit and site order."""
    key, value = "SF_MPS_GAUGE", ("1" if gauge else "0")
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        state = sf.cat(circuit, mode="mps", bond_dim=chi).simulate({})
        wire_order = np.asarray(state.numpy()).ravel()
        perm = [int(p) for p in state.perm()]
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    return {
        "qubit": wire_order,
        "site": _site_view(wire_order, perm),
        "perm": perm,
        "report": state.truncation_report(),
    }


@functools.lru_cache(maxsize=1)
def _fixture():
    """Exact reference plus every gauge/stride combination, computed once."""
    out = {}
    for stride in STRIDES:
        circuit = _circuit(N, CYCLES, stride)
        ref = np.asarray(
            circuit.to_ir().simulate_to_state("statevector").numpy()).ravel()
        out[stride] = {
            "ref": ref,
            "on": _run(circuit, CHI, True),
            "off": _run(circuit, CHI, False),
        }
    return out


def test_cap_binds_so_the_fixture_means_something():
    fx = _fixture()
    for stride in STRIDES:
        loss = _infidelity(fx[stride]["on"]["qubit"], fx[stride]["ref"])
        assert loss > 1e-4, (
            f"chi={CHI} no longer binds on the stride={stride} fixture "
            f"(1-fid {loss:.3e}); the gauge comparison below is void"
        )


def test_gauge_still_pays_when_routed():
    fx = _fixture()
    for stride in STRIDES:
        ref = fx[stride]["ref"]
        on = _infidelity(fx[stride]["on"]["qubit"], ref)
        off = _infidelity(fx[stride]["off"]["qubit"], ref)
        assert off > MIN_GAIN * on, (
            f"stride={stride}: gauge off 1-fid {off:.3e} is not "
            f"{MIN_GAIN:.0f}x the gauged {on:.3e} -- the repair stopped "
            "reaching this path"
        )


def test_achieved_loss_is_invariant_under_the_wire_map():
    """The router moves wires; it must not change what gets truncated."""
    fx = _fixture()
    base = _infidelity(fx[STRIDES[0]]["on"]["qubit"], fx[STRIDES[0]]["ref"])
    for stride in STRIDES[1:]:
        loss = _infidelity(fx[stride]["on"]["qubit"], fx[stride]["ref"])
        assert abs(loss - base) <= 1e-6 * max(1.0, base), (
            f"stride={stride} truncates to 1-fid {loss:.6e} against {base:.6e} "
            "for the adjacent wiring; lazy routing is supposed to be "
            "accuracy-neutral, so this is a real loss of a guarantee"
        )


def test_perm_maps_a_routed_state_back_to_qubit_order():
    """`perm()` must exist and must still describe the routing."""
    fx = _fixture()
    ctl = fx[STRIDES[0]]["on"]
    assert ctl["perm"] == list(range(N)), (
        "the adjacency-only control must not permute its sites"
    )
    routed = fx[STRIDES[-1]]["on"]
    perm = routed["perm"]
    assert sorted(perm) == list(range(N)), f"perm is not a permutation: {perm}"
    assert perm != list(range(N)), (
        f"stride={STRIDES[-1]} must route (it moved 2q gates off the chain) "
        "yet perm is the identity -- the router never ran"
    )
    ref = fx[STRIDES[-1]]["ref"]
    # `numpy()` is qubit order, so it must match closely; un-routing it by
    # `perm` puts the amplitudes back on the tensor sites, which is a
    # relabeling of the same values and therefore near-orthogonal to `ref`.
    # A `perm` that stopped describing the routing would invert this pair.
    assert _infidelity(routed["qubit"], ref) < 1e-1
    assert _infidelity(routed["site"], ref) > 0.5


if __name__ == "__main__":
    fx = _fixture()
    for stride in STRIDES:
        ref = fx[stride]["ref"]
        on, off = fx[stride]["on"], fx[stride]["off"]
        f_on = _infidelity(on["qubit"], ref)
        f_off = _infidelity(off["qubit"], ref)
        print(f"stride={stride}  perm_identity={on['perm'] == list(range(N))}")
        print(f"   gauge ON  1-fid {f_on:.4e}  discard "
              f"{on['report']['discarded_weight']:.4e}")
        print(f"   gauge OFF 1-fid {f_off:.4e}  discard "
              f"{off['report']['discarded_weight']:.4e}  gain x{f_off / f_on:.2f}")
        print(f"   site-order 1-fid {_infidelity(on['site'], ref):.4e}")
