"""
Hardware layout selection and SWAP routing — the layout is *reported*, not dropped.

Why this file exists
--------------------
``sf.compile(circuit, level, target)`` used to hand the circuit to the Rust
pipeline, which ran SABRE and then threw the returned ``QubitMapping`` away
(``let (routed_dag, _mapping) = router.route(&result)``).  SABRE's job is to
*permute* logical qubits onto physical ones, so discarding the permutation
silently changes what the compiled circuit computes:

    cx(0, 2)  ->  CX(1, 2)          # HW-2, unitary overlap 0.500000

The wrapper then rebuilt the result on ``target.n_qubits`` wires, so a 5-qubit
input came back as an 84-qubit circuit (HW-4) with ``_metadata == {}``, i.e. no
record of the permutation that had been applied (HW-5).

What is done instead
--------------------
1. Pick an explicit initial layout: the first ``n`` nodes of a breadth-first
   walk of the coupling map.  Any BFS prefix is connected by construction (the
   node at position ``k`` has its BFS parent at an earlier position), so the
   region always supports routing, and for a device whose low-numbered qubits
   are connected it *is* the identity layout.
2. Route on that region with token swapping, tracking where every logical
   qubit is at every point of the circuit.
3. Restore the initial layout with the reverse of the SWAP sequence, so the
   compiled circuit ends with logical qubit ``k`` back on physical
   ``region[k]``.  Undoing the permutation is what makes the output unitary
   *equal* to the input unitary instead of a permuted copy of it.
4. Because the final arrangement equals the initial one, the physical circuit
   is the logical circuit with wire ``k`` renamed to ``region[k]``.  Renaming
   every wire back is a static relabelling, so the returned circuit is emitted
   directly on wires ``0 .. n-1`` — the same register as the input (HW-4) —
   and the layout that was used is stored in ``Circuit._metadata`` (HW-5).

Cost and opt-out
----------------
Restoring the layout costs one extra SWAP per swap the router inserted, so a
routed circuit contains twice as many SWAPs as a SABRE run that leaves the
qubits where they ended up.  ``restore_layout=False`` reproduces that cheaper
behaviour; the final permutation is still reported, so the caller can undo it
when interpreting results instead of paying for it in gates.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from superfermion.circuit import Circuit, GateRecord

__all__ = [
    "build_adjacency",
    "connected_region",
    "shortest_path",
    "graph_distances",
    "route_with_layout",
    "get_layout",
    "LAYOUT_METADATA_KEY",
]

#: Key under which :func:`route_with_layout` stores its report on the circuit.
LAYOUT_METADATA_KEY = "layout"

_SWAP = "SWAP"
#: Gates that name classical bits or carry no wire state of their own.
_STRUCTURAL = frozenset({"BARRIER", "MEASURE", "RESET"})


def build_adjacency(coupling_map: Sequence[Sequence[int]]) -> Dict[int, set]:
    """Undirected adjacency from a (possibly one-directional) edge list.

    Vendor calibration files usually list each coupler once, control first.
    Routing only needs to know that two qubits *can* interact, so the direction
    is dropped here.
    """
    adjacency: Dict[int, set] = {}
    for edge in coupling_map or ():
        try:
            a, b = int(edge[0]), int(edge[1])
        except (TypeError, IndexError, ValueError):
            continue
        if a == b:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def connected_region(
    n_qubits: int,
    adjacency: Dict[int, set],
    n_physical: Optional[int] = None,
) -> Optional[List[int]]:
    """First ``n_qubits`` nodes of a BFS walk, or ``None`` if there is no room.

    The returned list is a connected set of physical qubits, in BFS order, and
    doubles as the initial layout (``logical k -> region[k]``).
    """
    if n_qubits < 1:
        return None
    if n_physical is not None and n_qubits > n_physical:
        return None
    for root in sorted(adjacency):
        order: List[int] = []
        seen = {root}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in sorted(adjacency.get(node, ())):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        if len(order) >= n_qubits:
            return order[:n_qubits]
    return None


def shortest_path(
    adjacency: Dict[int, set], source: int, target: int
) -> Optional[List[int]]:
    """Shortest path from ``source`` to ``target`` inclusive, or ``None``."""
    if source == target:
        return [source]
    previous: Dict[int, Optional[int]] = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour in previous:
                continue
            previous[neighbour] = node
            if neighbour == target:
                path = [target]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path
            queue.append(neighbour)
    return None


def graph_distances(adjacency: Dict[int, set], n_qubits: int) -> List[List[float]]:
    """All-pairs shortest-path lengths (``inf`` where unreachable)."""
    unreachable = float("inf")
    rows: List[List[float]] = []
    for start in range(n_qubits):
        row = [unreachable] * n_qubits
        row[start] = 0.0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency.get(node, ()):
                if neighbour >= n_qubits or row[neighbour] != unreachable:
                    continue
                row[neighbour] = row[node] + 1.0
                queue.append(neighbour)
        rows.append(row)
    return rows


def _rename(record: GateRecord, wires: Sequence[int]) -> GateRecord:
    """Copy ``record`` onto different wires, keeping every other field."""
    return GateRecord(
        name=record.name,
        qubits=[int(q) for q in wires],
        params=list(record.params),
        classical_bits=list(record.classical_bits),
        condition=record.condition,
        matrix=record.matrix,
    )


def route_with_layout(
    circuit: Circuit,
    target: Any = None,
    *,
    coupling_map: Optional[Sequence[Sequence[int]]] = None,
    n_physical: Optional[int] = None,
    restore_layout: bool = True,
) -> Tuple[Circuit, Dict[str, Any]]:
    """Route ``circuit`` onto the target connectivity, returning the layout used.

    Args:
        circuit: Circuit to route. Its wires are logical qubit indices.
        target: Optional object with ``coupling_map`` / ``n_qubits`` attributes
            (a :class:`~superfermion.compiler.specs.HardwareSpec`). Ignored when
            ``coupling_map`` is given explicitly.
        coupling_map: Edge list, overriding ``target.coupling_map``.
        n_physical: Device width, overriding ``target.n_qubits``.
        restore_layout: Undo the permutation the router introduced (default).
            Disable to keep the cheaper SABRE-style output and undo the
            reported ``final_layout`` yourself.

    Returns:
        ``(routed_circuit, report)``. ``report["routed"]`` is ``False`` when the
        device offers no usable region; the circuit is then returned untouched
        so the caller can fall back to whatever it did before.
    """
    n_logical = circuit.n_qubits
    if coupling_map is None and target is not None:
        coupling_map = list(getattr(target, "coupling_map", []) or ())
    if n_physical is None:
        n_physical = int(getattr(target, "n_qubits", n_logical) or n_logical)

    report: Dict[str, Any] = {
        "routed": False,
        "n_logical_qubits": n_logical,
        "n_physical_qubits": int(n_physical) if n_physical else None,
        "initial_layout": None,
        "final_layout": None,
        "swaps_inserted": 0,
        "restore_swaps": 0,
        "unrouted_gates": [],
        "reason": None,
    }

    adjacency = build_adjacency(coupling_map or ())
    if not adjacency:
        report["reason"] = "target declares no coupling map"
        return circuit, report

    region = connected_region(n_logical, adjacency, n_physical)
    if region is None:
        report["reason"] = (
            f"the {n_physical}-qubit coupling map has no connected set of "
            f"{n_logical} qubits to hold the circuit"
        )
        return circuit, report

    # Work in region-local indices 0 .. n-1.  Those are exactly the wire labels
    # of the returned circuit, because region[k] is renamed back to k at the end.
    local = {physical: index for index, physical in enumerate(region)}
    sub: Dict[int, set] = {
        local[physical]: {
            local[nbr] for nbr in adjacency.get(physical, ()) if nbr in local
        }
        for physical in region
    }
    distances = graph_distances(sub, n_logical)

    at: List[int] = list(range(n_logical))   # position -> logical qubit
    pos: List[int] = list(range(n_logical))  # logical qubit -> position
    records: List[GateRecord] = []
    swap_log: List[Tuple[int, int]] = []

    def insert_swap(u: int, v: int) -> None:
        """Emit a SWAP between two positions and move the tokens with it."""
        token_u, token_v = at[u], at[v]
        at[u], at[v] = token_v, token_u
        pos[token_u], pos[token_v] = v, u
        records.append(GateRecord(name=_SWAP, qubits=[u, v]))
        swap_log.append((u, v))

    def restore() -> None:
        """Undo every SWAP inserted so far, putting each qubit back home."""
        # Replaying the recorded sequence backwards restores the layout by
        # construction.  Consecutive inverses cancel, which is the only pruning
        # that is always safe here.
        replay: List[Tuple[int, int]] = []
        for swap in reversed(swap_log):
            if replay and replay[-1] == swap:
                replay.pop()
            else:
                replay.append(swap)
        for u, v in replay:
            insert_swap(u, v)
        report["restore_swaps"] += len(replay)

    circuit._ensure_gates()
    gate_list = circuit._gates

    # A trailing block of measurements is *terminal sampling*, not a mid-circuit
    # collapse.  Putting SWAPs after it would turn those measurements into
    # mid-circuit ones and change what ``run()`` reports, so the layout is
    # restored just before that block instead of at the very end.
    tail_start = len(gate_list)
    while tail_start > 0 and gate_list[tail_start - 1].name.upper() in _STRUCTURAL:
        tail_start -= 1
    restored = False

    for index, record in enumerate(gate_list):
        if restore_layout and not restored and index == tail_start:
            if swap_log:
                restore()
            restored = True

        name = record.name.upper()
        wires = [int(q) for q in record.qubits]

        if name in _STRUCTURAL or not wires:
            records.append(_rename(record, [pos[q] for q in wires]))
            continue

        if len(wires) == 2:
            a, b = pos[wires[0]], pos[wires[1]]
            if b not in sub.get(a, ()):
                path = shortest_path(sub, a, b)
                if path is None:
                    report["unrouted_gates"].append(name)
                else:
                    # Bringing either token towards the other costs the same
                    # number of SWAPs, so pick the end position that leaves
                    # both tokens closest to home -- that is what the restore
                    # pass below has to pay for.
                    stay = (
                        distances[path[-2]][wires[0]] + distances[path[-1]][wires[1]]
                    )
                    approach = (
                        distances[path[0]][wires[0]] + distances[path[1]][wires[1]]
                    )
                    if approach < stay:
                        for k in range(len(path) - 1, 1, -1):
                            insert_swap(path[k], path[k - 1])
                    else:
                        for k in range(len(path) - 2):
                            insert_swap(path[k], path[k + 1])
            a, b = pos[wires[0]], pos[wires[1]]
            if b not in sub.get(a, ()):
                report["unrouted_gates"].append(name)
            records.append(_rename(record, [pos[q] for q in wires]))
            continue

        # Gates on three or more wires need a clique in the coupling map.  The
        # compiler decomposes them before routing, so this is only reachable
        # when ``route_with_layout`` is called directly.
        mapped = [pos[q] for q in wires]
        if any(
            mapped[j] not in sub.get(mapped[i], ())
            for i in range(len(mapped))
            for j in range(len(mapped))
            if i != j
        ):
            report["unrouted_gates"].append(name)
        records.append(_rename(record, mapped))

    if restore_layout and not restored and swap_log:
        restore()
    inserted = len(swap_log) - report["restore_swaps"]

    routed = Circuit(
        n_logical,
        n_cbits=circuit.n_cbits,
        name=getattr(circuit, "name", None),
    )
    routed.extend_raw(records)

    # ``pos[k]`` is the region-local position of logical qubit k, so mapping it
    # back through ``region`` gives the physical wire it ends up on.  With the
    # layout restored this equals ``initial_layout``; without it, it is the
    # permutation the caller must undo when reading results
    # (``final_layout[logical] -> physical``).
    report.update(
        {
            "routed": True,
            "initial_layout": list(region),
            "final_layout": [region[position] for position in pos],
            "swaps_inserted": inserted,
        }
    )
    routed._metadata[LAYOUT_METADATA_KEY] = report
    return routed, report


def get_layout(circuit: Circuit) -> Optional[Dict[str, Any]]:
    """Layout report stored on a compiled circuit, or ``None``.

    ``sf.compile(circuit, level, target)`` records the initial layout, the
    final layout and the number of inserted SWAPs here (HW-5).
    """
    metadata = getattr(circuit, "_metadata", None)
    if not isinstance(metadata, dict):
        return None
    report = metadata.get(LAYOUT_METADATA_KEY)
    return report if isinstance(report, dict) else None
