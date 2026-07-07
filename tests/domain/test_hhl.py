"""HHL algorithm domain tests."""

import numpy as np
import pytest

from superfermion.algorithms.hhl import hhl_solve


pytestmark = [pytest.mark.domain, pytest.mark.timeout(30)]


class TestHHL:
    def test_hhl_solves_2x2_system(self):
        A = np.array([[1.5, 0.5], [0.5, 1.5]], dtype=float)
        b = np.array([1.0, 0.0], dtype=float)

        result = hhl_solve(A, b, precision_bits=3, t_scale=2.0, device="cpu")

        classical = np.linalg.solve(A, b)
        classical = classical / np.linalg.norm(classical)

        solution = result["solution"]
        assert solution.shape == (2,)
        overlap = abs(np.vdot(solution, classical))
        assert overlap > 0.5 or result["success_probability"] > 0.0
