"""CLI Integration Tests — Test every CLI command (Phase 4.1f)."""
import json
import os
import subprocess
import sys
import tempfile
import pytest

SF_CLI = [sys.executable, "-m", "superfermion.cli", "--no-banner"]


def _run_cli(*args, expect_success=True):
    """Run sf CLI with given arguments and return result."""
    result = subprocess.run(
        SF_CLI + list(args),
        capture_output=True, text=True, timeout=60,
        encoding='utf-8', errors='replace',
    )
    if expect_success:
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result


def _create_test_circuit_file():
    """Create a temporary circuit JSON file."""
    import superfermion as sf
    c = sf.Circuit(2).h(0).cx(0, 1)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp.write(c.to_json())
    tmp.close()
    return tmp.name


class TestBasicCommands:
    """Basic info/version/backends commands."""

    def test_info(self):
        """sf info shows system info."""
        r = _run_cli("info")
        assert "Version" in r.stdout
        assert "Platform" in r.stdout

    def test_version(self):
        """sf version shows version string."""
        r = _run_cli("version")
        assert "superfermion" in r.stdout.lower()

    def test_backends(self):
        """sf backends lists available backends."""
        r = _run_cli("backends")
        assert any(b in r.stdout.lower() for b in ["simulator", "statevector", "jax", "rust", "mps"])

    def test_info_no_banner_flag(self):
        """--no-banner suppresses banner."""
        r = _run_cli("--no-banner", "version")
        # Banner text should not appear
        assert "████" not in r.stdout

    def test_no_command_shows_help(self):
        """Running sf with no command shows help."""
        result = subprocess.run(
            SF_CLI, capture_output=True, text=True, timeout=60,
        )
        # Should either succeed or show usage
        assert "usage:" in (result.stdout + result.stderr).lower() or result.returncode == 0


class TestRunCommand:
    """sf run command tests."""

    def test_run_circuit(self):
        """Run a Bell state circuit from JSON."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("run", circuit_file, "--shots", "256")
            assert "Results" in r.stdout or "Counts" in r.stdout
        finally:
            os.unlink(circuit_file)
            result_file = circuit_file.replace('.json', '_result.json')
            if os.path.exists(result_file):
                os.unlink(result_file)

    def test_run_missing_file(self):
        """Run with nonexistent file fails gracefully."""
        result = subprocess.run(
            SF_CLI + ["run", "nonexistent_file.json"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0 or "Error" in result.stdout or "error" in result.stdout.lower()


class TestBenchmarkCommand:
    """sf benchmark command tests."""

    def test_benchmark_scaling(self):
        """Benchmark scaling suite runs."""
        r = _run_cli("benchmark", "--qubits", "4", "--iterations", "5", "--suite", "scaling")
        assert "Benchmark" in r.stdout

    def test_benchmark_compilation(self):
        """Benchmark compilation suite runs."""
        r = _run_cli("benchmark", "--suite", "compilation", "--qubits", "6")
        assert "Level" in r.stdout


class TestCompileCommand:
    """sf compile command tests."""

    def test_compile_circuit(self):
        """Compile a Bell state circuit."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("compile", circuit_file, "--level", "2")
            assert "Compiled" in r.stdout
        finally:
            os.unlink(circuit_file)
            compiled_file = circuit_file.replace('.json', '_compiled.json')
            if os.path.exists(compiled_file):
                os.unlink(compiled_file)

    def test_compile_with_basis(self):
        """Compile with specific basis gates."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("compile", circuit_file, "--basis", "rx,rz,cx", "--level", "1")
            assert "Compiled" in r.stdout
        finally:
            os.unlink(circuit_file)
            compiled_file = circuit_file.replace('.json', '_compiled.json')
            if os.path.exists(compiled_file):
                os.unlink(compiled_file)

    def test_compile_missing_file(self):
        """Compile with missing file fails gracefully."""
        result = subprocess.run(
            SF_CLI + ["compile", "nonexistent.json"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0 or "Error" in (result.stdout + result.stderr)


class TestNoiseCommand:
    """sf noise command tests."""

    def test_noise_inspect(self):
        """Inspect ibm_eagle noise model."""
        r = _run_cli("noise", "--model", "ibm_eagle")
        assert "depolarizing" in r.stdout.lower() or "channel" in r.stdout.lower()

    def test_noise_ideal(self):
        """Inspect ideal noise model."""
        r = _run_cli("noise", "--model", "ideal")
        assert "Noise Model" in r.stdout

    def test_noise_unknown_model(self):
        """Unknown noise model shows error."""
        result = subprocess.run(
            SF_CLI + ["noise", "--model", "bogus_model"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0 or "Error" in (result.stdout + result.stderr)


class TestTranspileCommand:
    """sf transpile command tests."""

    def test_transpile_help(self):
        """Transpile shows correct help."""
        result = subprocess.run(
            SF_CLI + ["transpile", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0

    def test_transpile_missing_file(self):
        """Transpile with missing file fails."""
        result = subprocess.run(
            SF_CLI + ["transpile", "nonexistent.json"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0 or "Error" in (result.stdout + result.stderr)


class TestCircuitCommand:
    """sf circuit command tests."""

    def test_circuit_bell(self):
        """Build Bell state."""
        r = _run_cli("circuit", "bell", "2")
        assert "Bell" in r.stdout or "Qubits" in r.stdout

    def test_circuit_ghz(self):
        """Build GHZ state."""
        r = _run_cli("circuit", "ghz", "3")
        assert "GHZ" in r.stdout or "Qubits" in r.stdout

    def test_circuit_qft(self):
        """Build QFT."""
        r = _run_cli("circuit", "qft", "4")
        assert "QFT" in r.stdout or "Qubits" in r.stdout

    def test_circuit_random(self):
        """Build random circuit."""
        r = _run_cli("circuit", "random", "4", "--depth", "5", "--seed", "42")
        assert "Random" in r.stdout or "Qubits" in r.stdout

    def test_circuit_info(self):
        """Inspect circuit from JSON."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("circuit", "info", circuit_file)
            assert "Qubits" in r.stdout
            assert "Gates" in r.stdout
        finally:
            os.unlink(circuit_file)

    def test_circuit_draw(self):
        """Draw circuit diagram."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("circuit", "draw", circuit_file)
            assert "Diagram" in r.stdout or "q" in r.stdout.lower()
        finally:
            os.unlink(circuit_file)


class TestStatevectorCommand:
    """sf statevector command tests."""

    def test_statevector_probs(self):
        """Show probability distribution."""
        circuit_file = _create_test_circuit_file()
        try:
            r = _run_cli("statevector", circuit_file, "--format", "probs")
            assert "probability" in r.stdout.lower() or "|" in r.stdout
        finally:
            os.unlink(circuit_file)


class TestGradientCommand:
    """sf gradient command tests."""

    def test_gradient_parameter_shift(self):
        """Compute parameter-shift gradient."""
        import superfermion as sf
        c = sf.Circuit(1)
        c.ry(sf.param("t"), 0)
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write(c.to_json())
        tmp.close()
        try:
            r = _run_cli("gradient", tmp.name, "--params", "0.5",
                         "--method", "parameter_shift")
            assert "Gradient" in r.stdout or "d<O>" in r.stdout
        finally:
            os.unlink(tmp.name)

    def test_gradient_missing_params(self):
        """Gradient without --params fails."""
        result = subprocess.run(
            SF_CLI + ["gradient", "some_file.json", "--method", "parameter_shift"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0


class TestTrainCommand:
    """sf train command tests."""

    def test_train_vqe(self):
        """Train VQE model."""
        r = _run_cli("train", "--model", "vqe", "--qubits", "2", "--iterations", "10")
        assert "Training" in r.stdout or "Optimal" in r.stdout or "VQE" in r.stdout

    def test_train_qaoa(self):
        """Train QAOA model."""
        r = _run_cli("train", "--model", "qaoa", "--graph", "ring4", "--iterations", "10")
        assert "Training" in r.stdout or "Optimal" in r.stdout or "QAOA" in r.stdout

    def test_train_unknown_model(self):
        """Unknown model fails."""
        result = subprocess.run(
            SF_CLI + ["train", "--model", "bogus"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0


class TestConfigCommand:
    """sf config command tests."""

    def test_config_show(self):
        """Show configuration."""
        r = _run_cli("config", "show")
        assert "[backends]" in r.stdout or "[simulation]" in r.stdout

    def test_config_set(self):
        """Set a configuration value."""
        r = _run_cli("config", "set", "project.name", "test_project")
        assert "Config set" in r.stdout or "test_project" in r.stdout


class TestValidateCommand:
    """sf validate command tests."""

    def test_validate_runs(self):
        """Validate runs without fatal errors."""
        r = _run_cli("validate", expect_success=False)
        assert "VALIDATED" in r.stdout or "FAILED" in r.stdout or "PASS" in r.stdout


class TestQECCommand:
    """sf qec command tests."""

    def test_qec_steane(self):
        """QEC lifecycle on Steane code."""
        r = _run_cli("qec", "--code", "steane")
        assert "QEC code" in r.stdout

    def test_qec_surface(self):
        """QEC lifecycle on Surface code."""
        r = _run_cli("qec", "--code", "surface_2d")
        assert "QEC code" in r.stdout


class TestExistingCommands:
    """Re-test existing commands after enhancements."""

    def test_shor(self):
        """sf shor works."""
        r = _run_cli("shor", "0000,0010", "--N", "15", "--a", "7")
        assert "Shor" in r.stdout or "Analysis" in r.stdout

    def test_chemistry(self):
        """sf chemistry works."""
        r = _run_cli("chemistry", "H2", "--top", "4")
        assert "Hamiltonian" in r.stdout or "Terms" in r.stdout

    def test_qaoa_existing(self):
        """sf qaoa still works."""
        r = _run_cli("qaoa", "--graph", "ring4", "--p-layers", "1")
        assert "QAOA" in r.stdout or "MaxCut" in r.stdout


class TestHelpOutput:
    """All commands have help output."""

    COMMANDS = [
        "info", "version", "backends", "validate", "run", "benchmark",
        "shor", "vqe", "qaoa", "chemistry", "qec",
        "compile", "transpile", "noise", "gradient", "train",
        "circuit", "statevector", "qpu", "config",
    ]

    @pytest.mark.parametrize("command", COMMANDS)
    def test_command_help(self, command):
        """Each command has --help output."""
        result = subprocess.run(
            SF_CLI + [command, "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"{command} --help failed: {result.stderr}"
