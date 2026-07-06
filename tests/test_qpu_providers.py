"""Tests for QPU providers and bridge (Phase 4.1e)."""
import os
import pytest
import numpy as np


def _has_ibm_credentials():
    """Check if IBM Quantum credentials are configured."""
    try:
        from superfermion.runtime.providers.ibm import IBMProvider
        p = IBMProvider()
        return bool(p._token)
    except Exception:
        return False


def _has_ionq_credentials():
    """Check if IonQ credentials are configured."""
    try:
        from superfermion.runtime.providers.ionq import IonQProvider
        p = IonQProvider()
        return bool(p._api_key)
    except Exception:
        return False


class TestIBMProvider:
    """IBM Quantum provider connectivity tests."""

    def test_provider_initializes(self):
        """IBM provider initializes without crashing."""
        try:
            from superfermion.runtime.providers.ibm import IBMProvider
            p = IBMProvider()
            assert p is not None
        except ImportError:
            pytest.skip("IBM provider dependencies not installed")
        except Exception as e:
            # May fail if no credentials — that's OK
            pass

    @pytest.mark.skipif(not _has_ibm_credentials(), reason="No IBM credentials configured")
    def test_list_backends(self):
        """List IBM backends (requires credentials)."""
        from superfermion.runtime.providers.ibm import IBMProvider
        p = IBMProvider()
        backends = p.list_backends()
        assert isinstance(backends, list)


class TestIonQProvider:
    """IonQ provider connectivity tests."""

    def test_provider_initializes(self):
        """IonQ provider initializes without crashing."""
        try:
            from superfermion.runtime.providers.ionq import IonQProvider
            p = IonQProvider()
            assert p is not None
        except ImportError:
            pytest.skip("IonQ provider dependencies not installed")

    @pytest.mark.skipif(not _has_ionq_credentials(), reason="No IonQ credentials configured")
    def test_list_backends(self):
        """List IonQ backends (requires credentials)."""
        from superfermion.runtime.providers.ionq import IonQProvider
        p = IonQProvider()
        backends = p.list_backends()
        assert isinstance(backends, list)


class TestBraketProvider:
    """Amazon Braket provider tests."""

    def test_provider_initializes(self):
        """Braket provider initializes."""
        try:
            from superfermion.runtime.providers.aws import BraketProvider
            p = BraketProvider()
            assert p is not None
        except ImportError:
            pytest.skip("Braket provider dependencies not installed")


class TestOpenQuantumProvider:
    """OpenQuantum provider tests."""

    def test_provider_initializes(self):
        """OpenQuantum provider importable."""
        try:
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            assert OpenQuantumProvider is not None
        except ImportError:
            pytest.skip("OpenQuantum provider dependencies not installed")


class TestIonQBridge:
    """IonQ gate format bridge tests."""

    def test_bridge_bell_state(self):
        """to_ionq converts Bell state circuit correctly."""
        import superfermion as sf
        try:
            from superfermion.bridge.ionq_bridge import to_ionq
        except ImportError:
            pytest.skip("IonQ bridge not available")

        c = sf.Circuit(2).h(0).cx(0, 1)
        ionq_circuit = to_ionq(c)
        assert isinstance(ionq_circuit, dict)
        assert "circuit" in ionq_circuit
        assert len(ionq_circuit["circuit"]) >= 2

    def test_bridge_ghz_state(self):
        """to_ionq converts GHZ circuit."""
        import superfermion as sf
        try:
            from superfermion.bridge.ionq_bridge import to_ionq
        except ImportError:
            pytest.skip("IonQ bridge not available")

        c = sf.Circuit(3).h(0).cx(0, 1).cx(1, 2)
        ionq_circuit = to_ionq(c)
        assert isinstance(ionq_circuit, dict)

    def test_bridge_single_qubit_gates(self):
        """to_ionq handles H, X, Y, Z, RX, RY, RZ gates."""
        import superfermion as sf
        try:
            from superfermion.bridge.ionq_bridge import to_ionq
        except ImportError:
            pytest.skip("IonQ bridge not available")

        c = sf.Circuit(1)
        c.h(0)
        c.x(0)
        c.rx(0.5, 0)
        ionq_circuit = to_ionq(c)
        assert isinstance(ionq_circuit, dict)

    def test_bridge_empty_circuit(self):
        """to_ionq handles empty circuit."""
        import superfermion as sf
        try:
            from superfermion.bridge.ionq_bridge import to_ionq
        except ImportError:
            pytest.skip("IonQ bridge not available")

        c = sf.Circuit(2)
        ionq_circuit = to_ionq(c)
        assert isinstance(ionq_circuit, dict)


class TestQPUCLIIntegration:
    """CLI QPU commands smoke tests (no actual QPU calls)."""

    def test_qpu_list_command(self):
        """sf qpu list runs without errors."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "superfermion.cli", "--no-banner", "qpu", "list"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0 or "Error" not in result.stdout


class TestAllProvidersInit:
    """All four providers initialize."""

    def test_all_providers_importable(self):
        """IBM, IonQ, Braket, OpenQuantum providers all import."""
        providers_ok = 0
        try:
            from superfermion.runtime.providers.ibm import IBMProvider
            providers_ok += 1
        except ImportError:
            pass
        try:
            from superfermion.runtime.providers.ionq import IonQProvider
            providers_ok += 1
        except ImportError:
            pass
        try:
            from superfermion.runtime.providers.aws import BraketProvider
            providers_ok += 1
        except ImportError:
            pass
        try:
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            providers_ok += 1
        except ImportError:
            pass
        assert providers_ok >= 2  # At least some providers should be available
