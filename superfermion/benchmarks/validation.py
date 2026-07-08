"""
Benchmark validation — generic basis-gate and topology checks.

Used after transpilation to verify the compiled circuit only uses
gates from the target basis set and respects the coupling map.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


def validate_basis(ops: Dict[str, int], basis_gates: List[str],
                   allow: Set[str] | None = None) -> List[str]:
    """Check that all operations are in the allowed basis set.

    Returns a list of violation messages (empty = valid).
    """
    allowed = {g.upper() for g in basis_gates}
    allowed |= {"BARRIER", "MEASURE", "RESET"}
    if allow:
        allowed |= {g.upper() for g in allow}

    violations = []
    for op_name in ops:
        if op_name.upper() not in allowed:
            violations.append(
                f"Gate '{op_name}' (x{ops[op_name]}) not in basis {sorted(basis_gates)}"
            )
    return violations


def validate_topology(ops: Dict[str, int], coupling_map: List[Tuple[int, int]],
                      two_qubit_ops: Dict[str, List[Tuple[int, int]]] | None = None
                      ) -> List[str]:
    """Check that all 2Q gate edges exist in the coupling map.

    This is a structural check only — requires the caller to provide
    the actual qubit pairs used by each 2Q gate.
    """
    if not coupling_map:
        return []
    edge_set = set()
    for a, b in coupling_map:
        edge_set.add((a, b))
        edge_set.add((b, a))

    violations = []
    if two_qubit_ops:
        for gate_name, qubit_pairs in two_qubit_ops.items():
            for pair in qubit_pairs:
                if tuple(pair) not in edge_set:
                    violations.append(
                        f"Gate '{gate_name}' on qubits {pair} violates coupling map"
                    )
    return violations
