"""
Parameter utilities for variational quantum circuits.

Usage:
    >>> import superfermion as sf
    >>> circuit = sf.Circuit(1).rx(sf.param("theta"), 0)
"""

from __future__ import annotations


class SymbolicParameter:
    """A symbolic parameter for variational circuits.

    This parameter is resolved at execution time via circuit.bind().
    It enables the parameter-shift rule for gradient computation.

    Args:
        name: Human-readable name (e.g. "theta_0", "phi_layer2")

    Examples:
        >>> theta = sf.param("theta")
        >>> circuit = sf.Circuit(1).rx(theta, 0)
        >>> bound = circuit.bind({"theta": 1.57})
    """

    _counter: int = 0

    def __init__(self, name: str) -> None:
        self.name = name
        self.id = SymbolicParameter._counter
        SymbolicParameter._counter += 1

    def __repr__(self) -> str:
        return f"Parameter('{self.name}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicParameter):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


def param(name: str) -> SymbolicParameter:
    """Create a symbolic parameter for variational circuits.

    Args:
        name: Human-readable parameter name.

    Returns:
        SymbolicParameter that can be passed to rotation gates.

    Examples:
        >>> import superfermion as sf
        >>> theta = sf.param("theta")
        >>> circuit = sf.Circuit(1).rx(theta, 0)
    """
    return SymbolicParameter(name)
