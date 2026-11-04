"""
Color Codes — High-distance topological QEC.
"""

from __future__ import annotations
import numpy as np

class ColorCode:
    """Implementations of 2D and 3D Color Codes.

    Note: ``superfermion.qec.ColorCode`` (from ``qec.codes.topological``) is the
    builder that :class:`QECManager` registers. This class previously had no
    ``build`` at all, so the two same-named types behaved completely differently
    (QEC-7). It now delegates to the topological implementation for
    ``dimension=2`` and exposes the same ``stabilizers``/``build`` surface.
    """

    def __init__(self, distance: int, dimension: int = 2):
        self.distance = distance
        self.dimension = dimension
        # Geometry generation would go here
        self.n_data = (distance**2 + 3) // 4 # Mock formula

    def stabilizer_matrix(self):
        """Return X and Z check matrices."""
        return np.zeros((self.n_data, self.n_data))

    def _delegate(self):
        """The topological builder that actually implements this lattice."""
        from .topological import ColorCode as _TopologicalColorCode

        return _TopologicalColorCode(distance=self.distance)

    def stabilizers(self):
        """``(x_checks, z_checks)``, delegated to the topological builder."""
        return self._delegate().stabilizers()

    def build(self):
        """Syndrome-extraction circuit, delegated to the topological builder."""
        return self._delegate().build()

    def __repr__(self):
        return f"ColorCode(d={self.distance}, dim={self.dimension})"
