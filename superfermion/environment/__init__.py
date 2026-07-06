"""
Environment Module — Runtime detection and display adapters.
"""

from superfermion.environment.detect import (
    Environment,
    detect_environment,
    current_environment,
    get_display_backend,
    display_circuit_html,
    display_result_html,
)

__all__ = [
    "Environment",
    "detect_environment",
    "current_environment",
    "get_display_backend",
    "display_circuit_html",
    "display_result_html",
]
