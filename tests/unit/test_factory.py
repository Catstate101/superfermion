"""Unit tests for backend factory registration and lookup."""

import pytest

from superfermion.backends.base import Backend
from superfermion.backends.factory import get_backend, list_backends


pytestmark = pytest.mark.unit


class TestGetBackend:
    def test_statevector_returns_backend_instance(self):
        backend = get_backend("statevector")
        assert isinstance(backend, Backend)

    def test_none_auto_selects(self):
        backend = get_backend(None)
        assert isinstance(backend, Backend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_backend("nonexistent_backend_xyz")


class TestListBackends:
    def test_returns_list_of_strings(self):
        names = list_backends()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_core_backends_registered(self):
        names = list_backends()
        assert "statevector" in names
        assert "singularity" in names
        assert "stabilizer" in names
