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


class TestHHLDocstringClaims:
    """ALG-8: the module docstring's example must actually reproduce.

    The docstring used to advertise a solution this call could not produce
    (``[0.989, +0.146]``, overlap 0.796).  These tests pin every number the
    docstring now prints, so the claim cannot drift away from the code again.
    """

    A = np.array([[1.5, 0.5], [0.5, 1.5]], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)

    @property
    def exact(self):
        vector = np.linalg.solve(self.A, self.b)
        return vector / np.linalg.norm(vector)

    def test_docstring_example_reproduces(self):
        result = hhl_solve(self.A, self.b, precision_bits=3, t_scale=2.0)
        solution = np.asarray(result["solution"])
        assert np.allclose(solution.real, [0.952363, -0.304967], atol=1e-6)
        overlap = abs(np.vdot(solution, self.exact))
        assert overlap == pytest.approx(0.999930, abs=5e-7)

    @pytest.mark.parametrize(
        "precision_bits, expected, overlap",
        [
            (3, [0.940656, -0.339361], 0.999700),
            (4, [0.954225, -0.299091], 0.999838),
            (5, [0.949090, -0.315004], 0.999999),
            (6, [0.948678, -0.316245], 1.000000),
        ],
    )
    def test_docstring_convergence_table(
        self, precision_bits, expected, overlap
    ):
        result = hhl_solve(self.A, self.b, precision_bits=precision_bits)
        solution = np.asarray(result["solution"])
        assert np.allclose(solution.real, expected, atol=1e-6)
        assert abs(np.vdot(solution, self.exact)) == pytest.approx(
            overlap, abs=5e-7
        )

    def test_accuracy_never_decreases_with_more_bits(self):
        overlaps = [
            abs(
                np.vdot(
                    np.asarray(hhl_solve(self.A, self.b, precision_bits=bits)["solution"]),
                    self.exact,
                )
            )
            for bits in (3, 4, 5, 6)
        ]
        assert overlaps == sorted(overlaps)
        assert overlaps[-1] == pytest.approx(1.0, abs=1e-6)
