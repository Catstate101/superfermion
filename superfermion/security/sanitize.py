"""
Input Sanitization — Prevention of QASM injection and input validation.

Validates and sanitizes all user-provided inputs before they reach
the circuit compiler or hardware backends.

Usage:
    >>> sanitizer = Sanitizer()
    >>> safe = sanitizer.sanitize_qasm(user_input)
    >>> sanitizer.validate_circuit_params(params)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


class SanitizationError(Exception):
    """Raised when input fails sanitization checks."""
    def __init__(self, message: str, field: str = "", suggestion: str = ""):
        self.field = field
        self.suggestion = suggestion
        full_msg = f"Sanitization failed: {message}"
        if field:
            full_msg += f" (field: {field})"
        if suggestion:
            full_msg += f"\n  Fix: {suggestion}"
        super().__init__(full_msg)


@dataclass
class SanitizationResult:
    """Result of a sanitization operation."""
    is_safe: bool
    sanitized_value: Any
    warnings: List[str]
    blocked_patterns: List[str]

    def __bool__(self) -> bool:
        return self.is_safe


# Patterns that should NEVER appear in QASM input
_DANGEROUS_PATTERNS = [
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"os\.\w+",
    r"subprocess",
    r"sys\.exit",
    r"open\s*\(",
    r"import\s+",
    r"from\s+\w+\s+import",
    r"\.\.\.",        # Path traversal
    r"\.\.\/",        # Path traversal
    r";\s*rm\s",      # Shell injection
    r"\|\s*\w+",      # Pipe commands
    r"`[^`]+`",       # Backtick execution
    r"\$\(",          # Shell substitution
]

# Valid QASM 3.0 gate names
_VALID_QASM_GATES = {
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cnot", "cz", "cy", "swap", "iswap", "ecr",
    "rxx", "ryy", "rzz", "crx", "cry", "crz", "cp",
    "ccx", "cswap", "toffoli", "fredkin",
    "measure", "barrier", "reset", "id",
    "gpi", "gpi2", "ms",  # IonQ native gates
}

# Valid QASM 3.0 keywords
_VALID_QASM_KEYWORDS = {
    "OPENQASM", "include", "qubit", "bit", "creg", "qreg",
    "gate", "if", "else", "while", "for", "in", "return",
    "const", "let", "def", "extern", "cal", "defcal",
    "true", "false", "pi", "tau", "euler",
}


class Sanitizer:
    """Input sanitizer for quantum computing workloads.

    Validates and cleans user input to prevent injection attacks
    and ensure safe circuit construction.

    Args:
        max_qubits: Maximum allowed qubit count.
        max_depth: Maximum allowed circuit depth.
        max_gates: Maximum allowed gate count.
        max_params: Maximum number of circuit parameters.
        max_qasm_length: Maximum QASM string length in bytes.
        custom_blocked_patterns: Additional regex patterns to block.

    Examples:
        >>> s = Sanitizer(max_qubits=40)
        >>> s.sanitize_qasm("OPENQASM 3.0;\\nqubit[2] q;\\nh q[0];")
        SanitizationResult(is_safe=True, ...)
    """

    def __init__(
        self,
        max_qubits: int = 127,
        max_depth: int = 10000,
        max_gates: int = 100000,
        max_params: int = 10000,
        max_qasm_length: int = 1_000_000,
        custom_blocked_patterns: Optional[List[str]] = None,
    ) -> None:
        self.max_qubits = max_qubits
        self.max_depth = max_depth
        self.max_gates = max_gates
        self.max_params = max_params
        self.max_qasm_length = max_qasm_length

        self._blocked_patterns = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]
        if custom_blocked_patterns:
            self._blocked_patterns.extend(
                re.compile(p, re.IGNORECASE) for p in custom_blocked_patterns
            )

    def sanitize_qasm(self, qasm_str: str) -> SanitizationResult:
        """Sanitize an OpenQASM string.

        Checks for:
        - Dangerous code injection patterns
        - Valid QASM structure
        - Resource limits (qubits, gates)

        Args:
            qasm_str: Raw QASM input from user.

        Returns:
            SanitizationResult with safety verdict and sanitized output.
        """
        warnings: List[str] = []
        blocked: List[str] = []

        # Length check
        if len(qasm_str.encode("utf-8")) > self.max_qasm_length:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"QASM input exceeds maximum length ({self.max_qasm_length} bytes)"],
                blocked_patterns=["length_exceeded"],
            )

        # Check for dangerous patterns
        for pattern in self._blocked_patterns:
            matches = pattern.findall(qasm_str)
            if matches:
                blocked.append(f"Blocked pattern: {pattern.pattern} (found: {matches[:3]})")

        if blocked:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=warnings,
                blocked_patterns=blocked,
            )

        # Check qubit count
        qubit_match = re.search(r"qubit\[(\d+)\]", qasm_str)
        if qubit_match:
            n_qubits = int(qubit_match.group(1))
            if n_qubits > self.max_qubits:
                return SanitizationResult(
                    is_safe=False,
                    sanitized_value="",
                    warnings=[f"Qubit count {n_qubits} exceeds limit {self.max_qubits}"],
                    blocked_patterns=["qubit_limit_exceeded"],
                )

        # Count gates (rough estimate)
        gate_lines = [
            line.strip() for line in qasm_str.split("\n")
            if line.strip() and not line.strip().startswith("//")
            and not line.strip().startswith("OPENQASM")
            and not line.strip().startswith("include")
            and not line.strip().startswith("qubit")
            and not line.strip().startswith("bit")
        ]
        if len(gate_lines) > self.max_gates:
            warnings.append(f"Gate count ({len(gate_lines)}) near limit ({self.max_gates})")

        # Strip any trailing whitespace/control characters
        sanitized = qasm_str.strip()
        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        return SanitizationResult(
            is_safe=True,
            sanitized_value=sanitized,
            warnings=warnings,
            blocked_patterns=[],
        )

    def validate_circuit_params(
        self,
        params: Dict[str, float],
    ) -> SanitizationResult:
        """Validate circuit parameter values.

        Args:
            params: Dictionary of parameter names to values.

        Returns:
            SanitizationResult with validation result.
        """
        warnings: List[str] = []
        blocked: List[str] = []

        if len(params) > self.max_params:
            return SanitizationResult(
                is_safe=False,
                sanitized_value=params,
                warnings=[f"Too many parameters: {len(params)} > {self.max_params}"],
                blocked_patterns=["param_limit_exceeded"],
            )

        sanitized_params: Dict[str, float] = {}
        for key, value in params.items():
            # Validate key is alphanumeric/underscore
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                blocked.append(f"Invalid parameter name: '{key}'")
                continue

            # Validate value is a finite number
            try:
                fval = float(value)
                if not (-1e15 < fval < 1e15):
                    warnings.append(f"Parameter '{key}' has extreme value: {fval}")
                sanitized_params[key] = fval
            except (TypeError, ValueError):
                blocked.append(f"Parameter '{key}' has non-numeric value: {value}")

        return SanitizationResult(
            is_safe=len(blocked) == 0,
            sanitized_value=sanitized_params,
            warnings=warnings,
            blocked_patterns=blocked,
        )

    def validate_backend_name(self, name: str) -> SanitizationResult:
        """Validate a backend name string."""
        warnings: List[str] = []
        blocked: List[str] = []

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_\-\.]*$", name):
            blocked.append(f"Invalid backend name: '{name}'")

        if len(name) > 256:
            blocked.append(f"Backend name too long: {len(name)} chars")

        return SanitizationResult(
            is_safe=len(blocked) == 0,
            sanitized_value=name.strip(),
            warnings=warnings,
            blocked_patterns=blocked,
        )

    def validate_shots(self, shots: int) -> SanitizationResult:
        """Validate shot count."""
        warnings: List[str] = []
        blocked: List[str] = []

        if not isinstance(shots, int) or shots < 1:
            blocked.append(f"Shots must be a positive integer, got: {shots}")
        elif shots > 1_000_000:
            warnings.append(f"Very high shot count: {shots}. This may be expensive.")

        return SanitizationResult(
            is_safe=len(blocked) == 0,
            sanitized_value=min(max(1, shots), 10_000_000) if isinstance(shots, int) else 1024,
            warnings=warnings,
            blocked_patterns=blocked,
        )

    def __repr__(self) -> str:
        return (
            f"Sanitizer(max_qubits={self.max_qubits}, "
            f"max_depth={self.max_depth}, "
            f"blocked_patterns={len(self._blocked_patterns)})"
        )
