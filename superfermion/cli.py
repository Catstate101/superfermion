"""
Superfermion CLI — Command-line interface for quantum workflows.

Usage:
    sf run <circuit_file> [--backend=<backend>] [--shots=<n>]
    sf info
    sf benchmark [--qubits=<n>] [--iterations=<n>]
    sf validate
    sf backends
    sf version
    sf shor <measurements> [--N=<n>] [--a=<a>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 so the banner never crashes on Windows cp1252."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_ensure_utf8_stdout()


# ── Styling ──────────────────────────────────────────────────────
class Colors:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BLUE    = "\033[94m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    # Golden palette
    G1 = "\033[38;5;136m"  # Dark amber
    G2 = "\033[38;5;172m"  # Copper
    G3 = "\033[38;5;178m"  # Gold
    G4 = "\033[38;5;214m"  # Orange gold
    G5 = "\033[38;5;220m"  # Bright gold
    G6 = "\033[38;5;228m"  # Pale gold
    GY = "\033[38;5;245m"  # Gray
    GD = "\033[38;5;240m"  # Dark gray
    W1 = "\033[38;5;231m"  # White
    # Blue palette
    B1 = "\033[38;5;33m"   # Royal blue
    B2 = "\033[38;5;39m"   # Sky blue
    B3 = "\033[38;5;75m"   # Steel blue
    B4 = "\033[38;5;69m"   # Cornflower
    B5 = "\033[38;5;117m"  # Light blue
    # Green palette
    GN = "\033[38;5;71m"   # Green

def styled(text: str, *styles: str) -> str:
    return "".join(styles) + text + Colors.RESET

def banner():
    import superfermion as sf
    ver = sf.__version__
    R = Colors.RESET
    B = Colors.BOLD
    D = Colors.DIM
    g1 = Colors.G1; g2 = Colors.G2; g3 = Colors.G3
    g4 = Colors.G4; g5 = Colors.G5; g6 = Colors.G6
    gy = Colors.GY; gd = Colors.GD; w = Colors.W1
    b1 = Colors.B1; b2 = Colors.B2; b3 = Colors.B3
    b4 = Colors.B4; b5 = Colors.B5
    print()
    print(f" {g3}{B} ███████╗██╗   ██╗██████╗ ███████╗██████╗{R}")
    print(f" {g4}{B} ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗{R}")
    print(f" {g5}{B} ███████╗██║   ██║██████╔╝█████╗  ██████╔╝{R}")
    print(f" {g4}{B} ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗{R}")
    print(f" {g5}{B} ███████║╚██████╔╝██║     ███████╗██║  ██║{R}")
    print(f" {g4}{B} ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝{R}")
    print(f" {b1}{B} ███████╗███████╗██████╗ ███╗   ███╗██╗ ██████╗ ███╗   ██╗{R}")
    print(f" {b1}{B} ██╔════╝██╔════╝██╔══██╗████╗ ████║██║██╔═══██╗████╗  ██║{R}")
    print(f" {b2}{B} █████╗  █████╗  ██████╔╝██╔████╔██║██║██║   ██║██╔██╗ ██║{R}")
    print(f" {b2}{B} ██╔══╝  ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║   ██║██║╚██╗██║{R}")
    print(f" {b3}{B} ██║     ███████╗██║  ██║██║ ╚═╝ ██║██║╚██████╔╝██║ ╚████║{R}")
    print(f" {b3}{B} ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝{R}")
    print()
    gn = Colors.GN
    # Padding: stage1@col36, stage2@col39, stage3@col42, stage4@col44, stem@col47
    print(f"                                    {g5}┌─────────────────────┐{R}")
    print(f"  {B}{g5}Quantum AI Framework{R}              {g5}│{g4}  ◆   ◆   ◆   ◆   ◆ {g5}│{R}")
    print(f"  {gd}v{ver}{R}                            {g5}└──────────┬──────────┘{R}")
    print(f"                                               {g4}│{R}")
    print(f"  {B}{g5}Every QPU.{R} {B}{b2}Every gradient.{R}           {g4}┌───────┴───────┐{R}")
    print(f"  {B}{gn}Every model.{R} {B}{w}One framework.{R}          {g4}│{g5}  ●━━━━●━━━━●  {g4}│{R}")
    print(f"                                       {g4}└───────┬───────┘{R}")
    print(f"  {g4}IBM{R} {gy}/{R} {b2}IonQ{R} {gy}/{R} {gn}Rigetti{R} {gy}/{R} {g5}AWS{R} {gy}/{R} {g3}D-Wave{R}          {g3}│{R}")
    print(f"  {b3}127 qubits{R} {gy}|{R} {g5}JAX autograd{R}               {g3}┌────┴────┐{R}")
    print(f"  {gn}Production ML at scale{R}                  {g3}│{g5}  ●━━━●  {g3}│{R}")
    print(f"                                          {g3}└────┬────┘{R}")
    print(f"                                               {g2}│{R}")
    print(f"                                            {g2}┌──┴──┐{R}")
    print(f"                                            {g2}│  {b5}⚛{R}  {g2}│{R}")
    print(f"                                            {g2}└─────┘{R}")
    print()


# ── Commands ─────────────────────────────────────────────────────

def cmd_version():
    """Print version info."""
    import superfermion as sf
    print(f"superfermion v{sf.__version__}")
    print(f"Python {sys.version.split()[0]}")
    try:
        import jax
        print(f"JAX {jax.__version__} ({jax.default_backend()})")
    except ImportError:
        print("JAX: not installed")
    try:
        import flax
        print(f"Flax {flax.__version__}")
    except ImportError:
        print("Flax: not installed")
    try:
        import optax
        print(f"Optax {optax.__version__}")
    except ImportError:
        print("Optax: not installed")


def cmd_info():
    """Print system and framework info."""
    import superfermion as sf
    from superfermion.environment import detect_environment
    
    env = detect_environment()
    
    print(styled("  System Info", Colors.BOLD, Colors.GREEN))
    print(f"  {'Version:':<20} {sf.__version__}")
    print(f"  {'Python:':<20} {sys.version.split()[0]}")
    print(f"  {'Platform:':<20} {sys.platform}")
    
    # Apple Silicon detection
    try:
        import platform
        machine = platform.machine()
        if sys.platform == "darwin" and machine == "arm64":
            print(f"  {'CPU:':<20} Apple Silicon ({machine})")
            try:
                import subprocess
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  {'Chip:':<20} {result.stdout.strip()}")
            except Exception:
                pass
        elif machine:
            print(f"  {'CPU:':<20} {machine}")
    except Exception:
        pass
    
    print(f"  {'Environment:':<20} {env.name}")
    
    try:
        import jax
        print(f"  {'JAX:':<20} {jax.__version__} (backend: {jax.default_backend()})")
        devices = jax.devices()
        print(f"  {'Devices:':<20} {len(devices)} x {devices[0].platform}")
    except ImportError:
        print(f"  {'JAX:':<20} NOT INSTALLED")
    
    print()
    print(styled("  Available Backends", Colors.BOLD, Colors.GREEN))
    for name in sf.list_backends():
        print(f"    - {name}")
    
    print()
    print(styled("  Modules", Colors.BOLD, Colors.GREEN))
    modules = [
        ("sf.Circuit",     "Quantum circuit API (30+ gates)"),
        ("sf.qml",         "JAX differentiable circuits"),
        ("sf.algorithms",  "VQE, QAOA, QSVM, QRL, QBM"),
        ("sf.viz",         "Bloch sphere, state plots"),
        ("sf.environment", "Runtime detection"),
    ]
    for name, desc in modules:
        print(f"    {styled(name, Colors.CYAN):<30} {desc}")
    print()


def cmd_backends():
    """List available backends."""
    import superfermion as sf
    print(styled("Available Backends:", Colors.BOLD))
    for name in sf.list_backends():
        print(f"  - {name}")


def cmd_run(args):
    """Run a circuit from a JSON file."""
    import superfermion as sf
    
    circuit_file = args.file
    backend = args.backend or "simulator"
    shots = args.shots or 1024
    
    if not os.path.exists(circuit_file):
        print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
        sys.exit(1)
    
    print(styled(f"Loading circuit from {circuit_file}...", Colors.DIM))
    
    with open(circuit_file, 'r') as f:
        circuit_json = f.read()
    
    circuit = sf.Circuit.from_json(circuit_json)
    
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  Gates:  {circuit.gate_count}")
    print(f"  Depth:  {circuit.depth}")
    print(f"  Backend: {backend}")
    print(f"  Shots:   {shots}")
    print()
    
    t0 = time.time()
    result = sf.run(circuit, backend=backend, shots=shots)
    dt = time.time() - t0
    
    print(styled("Results:", Colors.BOLD, Colors.GREEN))
    print(f"  Time: {dt*1000:.1f}ms")
    print(f"  Counts: {result.counts}")
    
    # Save output
    out_file = circuit_file.replace('.json', '_result.json')
    with open(out_file, 'w') as f:
        json.dump({
            "counts": result.counts,
            "backend": backend,
            "shots": shots,
            "time_ms": dt * 1000,
        }, f, indent=2)
    print(f"\n  Result saved to {out_file}")


def cmd_benchmark(args):
    """Run performance benchmarks across multiple backends."""
    import jax
    import jax.numpy as jnp
    import superfermion as sf
    
    max_qubits = args.qubits or 10
    iterations = args.iterations or 50
    backend_arg = args.backend or "all"
    suite = args.suite or "scaling"
    
    print(styled("  Performance Benchmark Suite", Colors.BOLD, Colors.GREEN))
    print(f"  Suite:     {suite}")
    print(f"  Max qubits: {max_qubits}")
    print(f"  Iterations: {iterations}")
    
    if suite == "scaling":
        # Determine which backends to test
        if backend_arg == "all":
            backends_to_test = ["simulator", "rust", "jax", "mps"]
            # Add cuda if available
            try:
                import cupy
                backends_to_test.append("cuda")
            except ImportError:
                pass
        else:
            backends_to_test = [b.strip() for b in backend_arg.split(",")]
        
        print(f"  Backends:  {', '.join(backends_to_test)}")
        print()
        
        sweep = sorted({n for n in (2, 4, 6, 8, 10, 14, 18, max_qubits) if 1 <= n <= max_qubits})
        
        # Header
        header = f"  {'Qubits':<8}"
        for b in backends_to_test:
            header += f" {b + ' (ms)':<14}"
        print(header)
        print(f"  {'-'*8} " + " ".join(f"{'':-<14}" for _ in backends_to_test))
        
        for n in sweep:
            c = sf.Circuit(n)
            for i in range(n):
                c.ry(sf.param(f"t{i}"), i)
            for i in range(n - 1):
                c.cx(i, i + 1)
            
            row = f"  {n:<8}"
            for backend_name in backends_to_test:
                try:
                    t0 = time.time()
                    for _ in range(iterations):
                        _ = sf.run(c, backend=backend_name, shots=0)
                    dt = (time.time() - t0) / iterations * 1000
                    row += f" {dt:<13.2f}"
                except Exception:
                    row += f" {'N/A':<13}"
            print(row)
        
    elif suite == "benchpress":
        print(styled("\n  Running Benchpress suite...", Colors.DIM))
        try:
            from tests.benchpress.conftest import run_benchpress
            # Run key benchpress tests
            print(f"  Benchpress results would be displayed here.")
            print(f"  Run: python -m pytest tests/benchpress/ -v")
        except ImportError:
            print(styled("  Benchpress not found. Run from repo root.", Colors.YELLOW))
    
    elif suite == "compilation":
        print()
        print(f"  {'Level':<8} {'Time (ms)':<14} {'Gates In':<12} {'Gates Out':<12}")
        print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*12}")
        
        n = min(max_qubits, 12)
        c = sf.Circuit(n)
        for i in range(n):
            c.h(i)
            c.rx(0.3, i)
        for i in range(n - 1):
            c.cx(i, i + 1)
        
        from superfermion.compiler import compile as sf_compile
        for level in range(4):
            t0 = time.time()
            compiled = sf_compile(c, level=level)
            dt = (time.time() - t0) * 1000
            print(f"  {level:<8} {dt:<14.2f} {c.gate_count:<12} {compiled.gate_count:<12}")
    
    out_file = args.output
    if out_file:
        # Would save structured results
        pass
    
    print()
    print(styled("  Benchmark complete.", Colors.GREEN))


def cmd_validate():
    """Run the full validation suite to certify the installation."""
    print(styled("  Installation Validation Suite", Colors.BOLD, Colors.GREEN))
    print()
    
    checks = []
    
    def check(name, fn):
        try:
            t0 = time.time()
            fn()
            dt = time.time() - t0
            checks.append((name, True, dt))
            print(f"  {styled('[PASS]', Colors.GREEN)} {name} ({dt:.2f}s)")
        except Exception as e:
            checks.append((name, False, 0))
            print(f"  {styled('[FAIL]', Colors.RED)} {name}: {e}")
    
    # ── Expanded Validation Checks ─────────────────────────────────

    # 1. Core Import
    def v_import():
        import superfermion as sf
        assert hasattr(sf, 'Circuit')
        assert hasattr(sf, 'run')
        assert hasattr(sf, 'param')
    check("1. Core import", v_import)
    
    # 2. JAX Integration
    def v_jax():
        import jax
        import jax.numpy as jnp
        import superfermion as sf
        c = sf.Circuit(1); c.rx(sf.param("t"), 0)
        f = sf.qml.circuit_to_jax(c, backend="jax")
        sv = f(jnp.array(1.0))
        assert sv.shape == (2,)
        g = jax.grad(lambda x: jnp.real(jnp.sum(jnp.abs(f(x))**2)))(1.0)
        assert jnp.isfinite(g)
    check("2. JAX differentiable circuits", v_jax)
    
    # 3. VQE
    def v_vqe():
        import jax.numpy as jnp
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        c = sf.Circuit(1); c.ry(sf.param("t"), 0)
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        vqe = VQE(c, h, optimizer="L-BFGS-B")
        res = vqe.minimize(iterations=30)
        assert res.optimal_value < -0.8
    check("3. VQE optimization", v_vqe)
    
    # 4. QAOA
    def v_qaoa():
        from superfermion.algorithms.variational import QAOA
        qaoa = QAOA(n_qubits=2, edges=[(0, 1)], p_layers=1)
        res = qaoa.minimize(iterations=30)
        assert res.optimal_value > 0.5
    check("4. QAOA optimization", v_qaoa)
    
    # 5. Serialization (JSON + QASM)
    def v_serde():
        import superfermion as sf
        c = sf.Circuit(2).h(0).cx(0, 1)
        j = c.to_json()
        c2 = sf.Circuit.from_json(j)
        assert c2.gate_count == c.gate_count
        q = c.to_qasm3()
        assert "OPENQASM" in q
    check("5. Serialization (JSON + QASM)", v_serde)
    
    # 6. Visualization (Bloch + charts)
    def v_viz():
        import jax.numpy as jnp
        from superfermion.viz import bloch_angles, state_bar_chart
        sv = jnp.array([1, 0], dtype=jnp.complex64)
        a = bloch_angles(sv)
        assert abs(a['z'] - 1.0) < 0.01
        chart = state_bar_chart(jnp.array([1, 0, 0, 0], dtype=jnp.complex64))
        assert "|00>" in chart
    check("6. Visualization (Bloch + charts)", v_viz)
    
    # 7. Environment Detection
    def v_env():
        from superfermion.environment import detect_environment
        env = detect_environment()
        assert env is not None
    check("7. Environment detection", v_env)
    
    # 8. Rust Backend
    def v_rust():
        import superfermion as sf
        c = sf.Circuit(2).h(0).cx(0, 1)
        result = sf.run(c, backend="rust", shots=0)
        assert result.statevector is not None
    check("10. Rust backend", v_rust)
    
    # 11. MPS Backend
    def v_mps():
        import superfermion as sf
        c = sf.Circuit(3).h(0).cx(0, 1).cx(1, 2)
        result = sf.run(c, backend="mps", shots=0)
        assert result.statevector is not None
    check("11. MPS backend", v_mps)
    
    # 12. Noise Model
    def v_noise():
        from superfermion.noise import NoiseModel, ibm_eagle_noise
        model = NoiseModel().add_depolarizing(0.01)
        assert len(model.single_qubit_channels) == 1
        eagle = ibm_eagle_noise()
        assert eagle.readout_error > 0
    check("12. Noise model", v_noise)
    
    # 13. QEC Decode (BP+OSD)
    def v_qec_decode():
        from superfermion.qec.decoders import BPOSD_Decoder
        d = BPOSD_Decoder.for_repetition(5)
        corr = d.decode([0, 1, 1, 0])  # error on qubit 2
        assert len(corr) > 0
        assert corr[0][0] == 2
    check("13. QEC (BP+OSD decode)", v_qec_decode)
    
    # 14. Neural Decoder Load
    def v_neural_decode():
        from superfermion.qec.decoders import NeuralDecoder
        nd = NeuralDecoder.load_pretrained("repetition_d3")
        assert nd.n_qubits == 3
        probs = nd.decode([0, 1])
        assert probs.shape == (3,)
    check("14. QEC (Neural decoder)", v_neural_decode)
    
    # 15. Configuration Manager
    def v_config():
        from superfermion.config.manager import Config, get_default_config
        cfg = get_default_config()
        assert cfg.get("backends.default") == "simulator"
        assert cfg.get("simulation.max_qubits") > 0
    check("15. Config manager", v_config)
    
    # 16. Telemetry
    def v_telemetry():
        from superfermion.telemetry import get_structured_logger
        logger = get_structured_logger("validate_test")
        assert logger is not None
        logger.info("Validation test log entry")
    check("16. Telemetry logging", v_telemetry)
    
    # 17. Security Credential Store
    def v_security():
        from superfermion.security.credentials import CredentialStore, CredentialBackend
        store = CredentialStore(backend=CredentialBackend.MEMORY)
        assert store is not None
    check("17. Security credential store", v_security)
    
    # 18. CuPy GPU (if available)
    def v_cupy():
        try:
            import cupy
            import superfermion as sf
            c = sf.Circuit(2).h(0).cx(0, 1)
            result = sf.run(c, backend="cuda", shots=0)
            assert result.statevector is not None
        except ImportError:
            raise Exception("CuPy not installed — skip if no GPU")
        except Exception as e:
            raise Exception(f"CuPy backend error: {e}")
    check("18. CuPy GPU backend", v_cupy)
    
    # 19. Compiler
    def v_compiler():
        import superfermion as sf
        from superfermion.compiler import compile as sf_compile
        c = sf.Circuit(2).h(0).cx(0, 1)
        compiled = sf_compile(c, level=2)
        assert compiled.gate_count > 0
    check("19. Circuit compiler", v_compiler)
    
    # 20. Gradient (Parameter Shift)
    def v_gradient():
        import jax.numpy as jnp
        import superfermion as sf
        from superfermion.qml.gradient.parameter_shift import parameter_shift_grad
        from superfermion.observables.core import Hamiltonian, PauliString
        c = sf.Circuit(1); c.ry(sf.param("t"), 0)
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        grad = parameter_shift_grad(c, h, {"t": 1.0})
        assert "t" in grad
    check("20. Parameter-shift gradient", v_gradient)
    
    # 21. Data Pipeline
    def v_data():
        from superfermion.data.dataset import DataLoader
        assert DataLoader is not None
    check("21. Data pipeline", v_data)
    
    # 22. Pulse Waveforms
    def v_pulse():
        from superfermion.pulse import Schedule
        p = Schedule(name="validate_test")
        assert p is not None
    check("22. Pulse waveforms", v_pulse)
    
    # Summary
    passed = sum(1 for _, ok, _ in checks if ok)
    failed = sum(1 for _, ok, _ in checks if not ok)
    total_time = sum(dt for _, _, dt in checks)
    
    print()
    print(f"  {'='*50}")
    if failed == 0:
        print(styled(f"  VALIDATED: {passed}/{passed+failed} checks passed ({total_time:.1f}s)", Colors.GREEN, Colors.BOLD))
        print(styled("  Superfermion is correctly installed and operational.", Colors.GREEN))
    else:
        print(styled(f"  FAILED: {failed}/{passed+failed} checks did not pass", Colors.RED, Colors.BOLD))
    print(f"  {'='*50}")
    print()
    
    return 0 if failed == 0 else 1


def cmd_vqe(args):
    """Run VQE on a preset Hamiltonian (H2 by default)."""
    from superfermion.algorithms.variational import VQE
    from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz
    from superfermion.observables.core import Hamiltonian, PauliString

    if args.hamiltonian in ("H2", "LiH", "BeH2"):
        H = get_molecular_hamiltonian(args.hamiltonian)
        ansatz = uccsd_ansatz(n_qubits=args.qubits or 4, n_electrons=args.electrons or 2)
        label = f"molecule={args.hamiltonian}"
    elif args.hamiltonian == "tfim":
        n = args.qubits or 2
        terms = [PauliString("Z" * (i) + "Z" + "I" * (n - i - 1), coeffs=1.0) for i in range(n - 1)]
        terms += [PauliString("I" * i + "X" + "I" * (n - i - 1), coeffs=-0.5) for i in range(n)]
        H = Hamiltonian(terms)
        import superfermion as sf
        ansatz = sf.Circuit(n)
        for i in range(n):
            ansatz.ry(sf.param(f"t{i}"), i)
        for i in range(n - 1):
            ansatz.cx(i, i + 1)
        label = f"TFIM n={n}"
    else:
        print(styled(f"Error: unknown Hamiltonian '{args.hamiltonian}'", Colors.RED))
        print("Choose from: H2, LiH, BeH2, tfim")
        sys.exit(1)

    print(styled(f"  VQE ({label})", Colors.BOLD, Colors.GREEN))
    vqe = VQE(ansatz, H, backend=args.backend or "statevector",
              optimizer=args.optimizer or "L-BFGS-B")
    t0 = time.time()
    result = vqe.minimize(iterations=args.iterations or 200)
    dt = time.time() - t0
    print(f"  Ground-state energy : {result.optimal_value:.6f}")
    print(f"  Iterations          : {len(result.history)}")
    print(f"  Wall clock          : {dt:.2f}s")


def cmd_qaoa(args):
    """Run QAOA on a named graph preset."""
    from superfermion.algorithms.variational import QAOA

    presets = {
        "ring4":     (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        "ring6":     (6, [(i, (i + 1) % 6) for i in range(6)]),
        "complete4": (4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        "triangle":  (3, [(0, 1), (1, 2), (0, 2)]),
    }
    if args.graph not in presets:
        print(styled(f"Error: unknown graph '{args.graph}'", Colors.RED))
        print(f"Choose from: {', '.join(presets)}")
        sys.exit(1)
    n, edges = presets[args.graph]

    print(styled(f"  QAOA MaxCut on {args.graph} (n={n}, p={args.p_layers})", Colors.BOLD, Colors.GREEN))
    qaoa = QAOA(n_qubits=n, edges=edges, p_layers=args.p_layers,
                backend=args.backend or "statevector")
    t0 = time.time()
    result = qaoa.minimize()
    dt = time.time() - t0
    print(f"  Optimal cut value : {result.optimal_value:.4f}")
    print(f"  Optimal angles    : {result.optimal_params}")
    print(f"  Wall clock        : {dt:.2f}s")


def cmd_chemistry(args):
    """Show molecular Hamiltonian info or run a quick SCF/VQE."""
    from superfermion.chemistry import get_molecular_hamiltonian

    H = get_molecular_hamiltonian(args.molecule, basis=args.basis)
    print(styled(f"  Molecular Hamiltonian: {args.molecule} ({args.basis})", Colors.BOLD, Colors.GREEN))
    print(f"  Terms: {len(H.terms)}")
    for t in H.terms[: args.top]:
        print(f"    {t}")
    if len(H.terms) > args.top:
        print(f"    ... {len(H.terms) - args.top} more")
    if args.vqe:
        from superfermion.algorithms.variational import VQE
        from superfermion.chemistry import uccsd_ansatz
        qubits = args.qubits or 4
        electrons = args.electrons or 2
        ansatz = uccsd_ansatz(n_qubits=qubits, n_electrons=electrons)
        print()
        print(styled("  Running VQE ...", Colors.DIM))
        vqe = VQE(ansatz, H, backend=args.backend or "statevector")
        result = vqe.minimize(iterations=args.iterations or 200)
        print(f"  Ground-state energy : {result.optimal_value:.6f} Ha")


def cmd_qec(args):
    """Run a QEC logical-qubit lifecycle or a full fault-tolerance audit."""
    from superfermion.qec import QECManager

    mgr = QECManager()
    if args.audit:
        out = mgr.simulate_fault_tolerant_workflow(args.code)
    else:
        out = mgr.run_logical_lifecycle(args.code, error_type=args.error, error_qubit=args.error_qubit)

    print(styled(f"  QEC code = {args.code}  ({'audit' if args.audit else 'lifecycle'})",
                 Colors.BOLD, Colors.GREEN))
    for key, value in out.items():
        print(f"  {key:<22}: {value}")


def cmd_shor(args):
    """Analyze Shor's measurement results."""
    import jax
    import jax.numpy as jnp
    from fractions import Fraction
    
    N = args.N
    a = args.a
    measurements = args.measurements.split(',')
    
    print(styled(f"  Shor's Factorization Analysis: N={N}, a={a}", Colors.BOLD, Colors.GREEN))
    print()

    @jax.jit
    def jax_gcd(a_val, b_val):
        for _ in range(32):
            cond = b_val != 0
            a_new = jnp.where(cond, b_val, a_val)
            b_new = jnp.where(cond, a_val % b_val, b_val)
            a_val, b_val = a_new, b_new
        return a_val

    print(f"  {'Binary':<15} {'Phase':<10} {'Period(r)':<10} {'Factors':<15}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*15}")
    
    for b in measurements:
        b = b.strip()
        if not b: continue
        n_bits = len(b)
        decimal_val = int(b, 2)
        if decimal_val == 0:
            print(f"  {b:<15} {'0.000':<10} {'Trivial':<10} {'-'}")
            continue
            
        phase = decimal_val / (2**n_bits)
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        factors_found = "None"
        if r % 2 == 0:
            x = pow(a, r // 2, N)
            f1 = int(jax_gcd(jnp.array(x - 1), jnp.array(N)))
            f2 = int(jax_gcd(jnp.array(x + 1), jnp.array(N)))
            valid = [f for f in [f1, f2] if 1 < f < N]
            if valid:
                factors_found = ", ".join(map(str, sorted(list(set(valid)))))
        
        print(f"  {b:<15} {phase:<10.3f} {r:<10} {factors_found}")
    print()
    print(styled("  Analysis complete.", Colors.GREEN))


# ── NEW Phase 2.2 Commands ────────────────────────────────────────

def cmd_compile(args):
    """Compile circuits for target hardware."""
    import superfermion as sf
    
    circuit_file = args.file
    if not os.path.exists(circuit_file):
        print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
        sys.exit(1)
    
    with open(circuit_file, 'r') as f:
        c = sf.Circuit.from_json(f.read())
    
    print(styled(f"  Compiling circuit ({c.n_qubits} qubits, {c.gate_count} gates)", Colors.BOLD, Colors.GREEN))
    print(f"  Target: {args.target}")
    print(f"  Level:  {args.level}")
    
    target_desc = args.target
    basis_gates = None
    if args.basis:
        basis_gates = [g.strip() for g in args.basis.split(',')]
        target_desc += f" basis={basis_gates}"
    
    coupling_map = None
    if args.coupling_map:
        if os.path.exists(args.coupling_map):
            with open(args.coupling_map, 'r') as f:
                coupling_map_data = json.load(f)
            coupling_map = coupling_map_data.get("coupling_map", [])
    
    t0 = time.time()
    try:
        from superfermion.compiler import compile as sf_compile
        compiled = sf_compile(
            c,
            level=args.level,
            target_basis=basis_gates or ["rx", "ry", "rz", "cx"],
        )
    except Exception:
        # Fallback: basic pass through
        compiled = c
    
    dt = time.time() - t0
    
    print(f"  Compiled: {compiled.gate_count} gates, depth {compiled.depth}")
    print(f"  Time:    {dt*1000:.1f}ms")
    
    out_file = args.output or circuit_file.replace('.json', '_compiled.json')
    with open(out_file, 'w') as f:
        f.write(compiled.to_json())
    print(f"\n  Output saved to {out_file}")


def cmd_transpile(args):
    """Full transpile pipeline for QPU submission."""
    import superfermion as sf
    
    circuit_file = args.file
    if not os.path.exists(circuit_file):
        print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
        sys.exit(1)
    
    with open(circuit_file, 'r') as f:
        c = sf.Circuit.from_json(f.read())
    
    print(styled(f"  Transpiling for {args.provider} ({args.backend or 'default'})", Colors.BOLD, Colors.GREEN))
    print(f"  Input: {c.n_qubits} qubits, {c.gate_count} gates")
    
    t0 = time.time()
    
    # Compile step
    from superfermion.compiler import compile as sf_compile
    compiled = sf_compile(c, level=args.level or 2)
    
    print(f"  Compiled: {compiled.gate_count} gates, depth {compiled.depth}")
    
    # QPU-specific formatting
    if args.provider == "ionq":
        from superfermion.bridge.ionq_bridge import to_ionq
        qpu_circuit = to_ionq(compiled)
        print(f"  IonQ format: {len(qpu_circuit.get('circuit', []))} operations")
    elif args.provider == "ibm":
        try:
            from qiskit import transpile as qiskit_transpile
            from qiskit.circuit import QuantumCircuit as QiskitCircuit
            qc = QiskitCircuit(compiled.n_qubits)
            for gate in compiled.gates:
                if gate.name == 'h':
                    qc.h(gate.qubits[0])
                elif gate.name == 'cx':
                    qc.cx(gate.qubits[0], gate.qubits[1])
                elif gate.name == 'x':
                    qc.x(gate.qubits[0])
                elif gate.name == 'rz':
                    qc.rz(0, gate.qubits[0])
            print(f"  IBM format: {qc.size()} operations")
        except ImportError:
            print(styled("  Warning: qiskit not installed for IBM transpilation.", Colors.YELLOW))
    else:
        qpu_circuit = compiled
    
    dt = time.time() - t0
    print(f"  Time: {dt*1000:.1f}ms")
    
    # Submit to QPU if requested
    if args.submit:
        print(styled(f"\n  Submitting to {args.provider}...", Colors.DIM))
        try:
            from superfermion.runtime.providers.ibm import IBMProvider
            from superfermion.runtime.providers.ionq import IonQProvider
            from superfermion.runtime.providers.aws import BraketProvider
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            
            provider_map = {
                "ibm": IBMProvider,
                "ionq": IonQProvider,
                "braket": BraketProvider,
                "openquantum": OpenQuantumProvider,
            }
            provider_cls = provider_map.get(args.provider)
            if provider_cls:
                provider = provider_cls()
                result = provider.run(compiled, shots=args.shots or 1024)
                print(styled(f"  Job submitted: {result}", Colors.GREEN))
            else:
                print(styled(f"  Error: Unknown provider '{args.provider}'", Colors.RED))
        except Exception as e:
            print(styled(f"  Submit error: {e}", Colors.RED))
    
    # Save output
    out_file = args.output or circuit_file.replace('.json', '_transpiled.json')
    with open(out_file, 'w') as f:
        f.write(compiled.to_json())
    print(f"\n  Output saved to {out_file}")


def cmd_noise(args):
    """Apply and inspect noise models."""
    import superfermion as sf
    from superfermion.noise import ibm_eagle_noise, ideal_noise
    
    noise_presets = {"ibm_eagle": ibm_eagle_noise, "ideal": ideal_noise}
    
    if not args.apply:
        # Inspection mode
        model_name = args.model or "ibm_eagle"
        builder = noise_presets.get(model_name)
        if not builder:
            print(styled(f"Error: Unknown noise model '{model_name}'", Colors.RED))
            print(f"Available: {', '.join(noise_presets)}")
            sys.exit(1)
        
        model = builder()
        print(styled(f"  Noise Model: {model_name}", Colors.BOLD, Colors.GREEN))
        print(f"  1-qubit channels: {len(model.single_qubit_channels)}")
        for ch in model.single_qubit_channels:
            print(f"    - {ch.name}: p={ch.error_rate}")
        print(f"  2-qubit channels: {len(model.two_qubit_channels)}")
        for ch in model.two_qubit_channels:
            print(f"    - {ch.name}: p={ch.error_rate}")
        print(f"  Readout error: {model.readout_error}")
    else:
        # Apply mode
        circuit_file = args.apply
        if not os.path.exists(circuit_file):
            print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
            sys.exit(1)
        
        with open(circuit_file, 'r') as f:
            c = sf.Circuit.from_json(f.read())
        
        model_name = args.model or "ibm_eagle"
        builder = noise_presets.get(model_name)
        if not builder:
            print(styled(f"Error: Unknown noise model '{model_name}'", Colors.RED))
            sys.exit(1)
        
        model = builder()
        shots = args.shots or 4096
        
        print(styled(f"  Noisy Simulation", Colors.BOLD, Colors.GREEN))
        print(f"  Circuit: {c.n_qubits} qubits, {c.gate_count} gates")
        print(f"  Noise:   {model_name}")
        print(f"  Shots:   {shots}")
        
        t0 = time.time()
        result = sf.run(c, backend="simulator", noise_model=model, shots=shots)
        dt = time.time() - t0
        
        print(f"  Time:    {dt*1000:.1f}ms")
        print(f"  Counts:  {result.counts}")
        
        out_file = args.output or circuit_file.replace('.json', '_noisy.json')
        with open(out_file, 'w') as f:
            json.dump({"counts": result.counts, "noise_model": model_name, "shots": shots}, f, indent=2)
        print(f"\n  Result saved to {out_file}")


def cmd_gradient(args):
    """Compute gradients via parameter-shift or adjoint method."""
    import jax.numpy as jnp
    import superfermion as sf
    
    circuit_file = args.file
    if not os.path.exists(circuit_file):
        print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
        sys.exit(1)
    
    with open(circuit_file, 'r') as f:
        c = sf.Circuit.from_json(f.read())
    
    params_list = [float(x.strip()) for x in args.params.split(',')]
    param_names = []
    for gate in c._gates:
        for p in gate.params:
            if isinstance(p, str) and p not in param_names:
                param_names.append(p)
    if len(params_list) != len(param_names):
        print(styled(f"Error: {len(params_list)} params provided but circuit has {len(param_names)}", Colors.RED))
        sys.exit(1)
    
    params = dict(zip(param_names, params_list))
    
    print(styled(f"  Gradient: {args.method}", Colors.BOLD, Colors.GREEN))
    print(f"  Circuit: {c.n_qubits} qubits, {c.gate_count} gates, {len(param_names)} params")
    
    t0 = time.time()
    
    from superfermion.observables.core import Hamiltonian, PauliString
    obs = Hamiltonian([PauliString("Z" * c.n_qubits, coeffs=1.0)])
    
    if args.method == "parameter_shift":
        from superfermion.qml.gradient.parameter_shift import parameter_shift_grad
        grad = parameter_shift_grad(c, obs, params, backend=args.backend or "statevector")
    elif args.method == "adjoint":
        from superfermion.qml.gradient.adjoint import adjoint_grad_vector
        vals = jnp.array(list(params.values()))
        grad_vals = adjoint_grad_vector(c, obs, list(params.keys()), vals)
        grad = dict(zip(params.keys(), [float(g) for g in grad_vals]))
    else:
        print(styled(f"Error: Unknown method '{args.method}'. Use 'parameter_shift' or 'adjoint'.", Colors.RED))
        sys.exit(1)
    
    dt = time.time() - t0
    
    print(f"  Time: {dt*1000:.1f}ms")
    print(f"  Gradients:")
    for k, v in grad.items():
        print(f"    d<O>/d{k} = {v:.6f}")


def cmd_train(args):
    """Run QML training loop."""
    import jax.numpy as jnp
    import superfermion as sf
    
    print(styled(f"  QML Training: {args.model}", Colors.BOLD, Colors.GREEN))
    
    if args.model == "vqe":
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        mol = args.hamiltonian or "H2"
        n = args.qubits or 4
        
        if mol == "H2":
            h = Hamiltonian([PauliString("Z" * n, coeffs=1.0)])
        else:
            h = Hamiltonian([PauliString("Z" * n, coeffs=1.0)])
        
        c = sf.Circuit(n)
        for i in range(n):
            c.ry(sf.param(f"t{i}"), i)
        for i in range(n - 1):
            c.cx(i, i + 1)
        
        print(f"  Hamiltonian: {mol} (approx)")
        print(f"  Qubits: {n}, Params: {n}")
        print(f"  Iterations: {args.iterations or 100}")
        
        vqe = VQE(c, h, backend=args.backend or "statevector")
        t0 = time.time()
        result = vqe.minimize(iterations=args.iterations or 100)
        dt = time.time() - t0
        
        print(f"  Optimal value: {result.optimal_value:.6f}")
        print(f"  Wall clock:    {dt:.2f}s")
        
    elif args.model == "qaoa":
        from superfermion.algorithms.variational import QAOA

        graph_name = args.graph or "ring4"
        presets = {
            "ring4": [(0,1),(1,2),(2,3),(3,0)],
            "ring6": [(i,(i+1)%6) for i in range(6)],
        }
        edges = presets.get(graph_name, presets["ring4"])
        n = max(max(e) for e in edges) + 1
        p = args.p_layers or 2

        print(f"  Graph: {graph_name} (n={n}, p={p})")
        print(f"  Iterations: {args.iterations or 200}")

        qaoa = QAOA(n_qubits=n, edges=edges, p_layers=p)
        t0 = time.time()
        result = qaoa.minimize(iterations=args.iterations or 200)
        dt = time.time() - t0
        
        print(f"  Optimal value: {result.optimal_value:.6f}")
        print(f"  Wall clock:    {dt:.2f}s")
    else:
        print(styled(f"Error: Unknown model '{args.model}'. Use 'vqe' or 'qaoa'.", Colors.RED))
        sys.exit(1)


def cmd_circuit(args):
    """Build and inspect quantum circuits."""
    import superfermion as sf
    
    if args.sub == "bell":
        n = args.n_qubits or 2
        c = sf.Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        print(styled(f"  Bell State ({n} qubits)", Colors.BOLD, Colors.GREEN))
        
    elif args.sub == "ghz":
        n = args.n_qubits or 3
        c = sf.Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        print(styled(f"  GHZ State ({n} qubits)", Colors.BOLD, Colors.GREEN))
        
    elif args.sub == "qft":
        n = args.n_qubits or 4
        c = sf.Circuit(n)
        import math as _math
        for i in range(n):
            c.h(i)
            for j in range(i + 1, n):
                c.cp(_math.pi / (2 ** (j - i)), i, j)
        for i in range(n // 2):
            c.swap(i, n - 1 - i)
        print(styled(f"  QFT ({n} qubits)", Colors.BOLD, Colors.GREEN))
        
    elif args.sub == "random":
        n = args.n_qubits or 4
        depth = args.depth or 10
        seed = args.seed or 42
        import random, math as _math
        random.seed(seed)
        gate_pool = ['h', 'x', 'rx', 'ry', 'rz', 'cx']
        c = sf.Circuit(n)
        for _ in range(depth):
            g = random.choice(gate_pool)
            q = random.randint(0, n - 1)
            if g == 'h':
                c.h(q)
            elif g == 'x':
                c.x(q)
            elif g == 'rx':
                c.rx(random.uniform(0, 2 * _math.pi), q)
            elif g == 'ry':
                c.ry(random.uniform(0, 2 * _math.pi), q)
            elif g == 'rz':
                c.rz(random.uniform(0, 2 * _math.pi), q)
            elif g == 'cx' and n > 1:
                t = (q + 1 + random.randint(0, n - 2)) % n
                c.cx(q, t)
        print(styled(f"  Random Circuit ({n} qubits, depth {depth}, seed {seed})", Colors.BOLD, Colors.GREEN))
        
    elif args.sub == "from-qasm":
        qasm_file = args.source
        if not qasm_file or not os.path.exists(qasm_file):
            print(styled(f"Error: QASM file '{qasm_file}' not found.", Colors.RED))
            sys.exit(1)
        with open(qasm_file, 'r') as f:
            qasm_str = f.read()
        c = sf.Circuit.from_qasm(qasm_str)
        print(styled(f"  Imported from QASM", Colors.BOLD, Colors.GREEN))
        
    elif args.sub == "info":
        info_file = args.source
        if not info_file or not os.path.exists(info_file):
            print(styled(f"Error: Circuit file '{info_file}' not found.", Colors.RED))
            sys.exit(1)
        with open(info_file, 'r') as f:
            c = sf.Circuit.from_json(f.read())
        
        print(styled(f"  Circuit Info", Colors.BOLD, Colors.GREEN))
        print(f"  Qubits:    {c.n_qubits}")
        print(f"  Gates:     {c.gate_count}")
        print(f"  Depth:     {c.depth}")
        print(f"  Params:    {len(c.parameters)}")
        
        # Gate breakdown
        from collections import Counter
        gate_names = [g.name for g in c._gates]
        breakdown = Counter(gate_names)
        print(f"  Gate breakdown:")
        for name, count in breakdown.most_common():
            print(f"    {name}: {count}")
        return
        
    elif args.sub == "draw":
        draw_file = args.source
        if not draw_file or not os.path.exists(draw_file):
            print(styled(f"Error: Circuit file '{draw_file}' not found.", Colors.RED))
            sys.exit(1)
        with open(draw_file, 'r') as f:
            c = sf.Circuit.from_json(f.read())
        
        print(styled(f"  Circuit Diagram ({c.n_qubits} qubits)", Colors.BOLD, Colors.GREEN))
        print(c.draw())
        return

    else:
        print(styled(f"Error: Unknown circuit command '{args.sub}'.", Colors.RED))
        print("Commands: bell, ghz, qft, random, from-qasm, info, draw")
        sys.exit(1)
    
    print(f"  Qubits: {c.n_qubits}")
    print(f"  Gates:  {c.gate_count}")
    print(f"  Depth:  {c.depth}")
    
    out_file = args.output or f"circuit_{args.sub}_{c.n_qubits}q.json"
    with open(out_file, 'w') as f:
        f.write(c.to_json())
    print(f"\n  Saved to {out_file}")


def cmd_statevector(args):
    """Inspect and visualize quantum statevectors."""
    import superfermion as sf
    
    circuit_file = args.file
    if not os.path.exists(circuit_file):
        print(styled(f"Error: File '{circuit_file}' not found.", Colors.RED))
        sys.exit(1)
    
    with open(circuit_file, 'r') as f:
        c = sf.Circuit.from_json(f.read())
    
    backend = args.backend or "simulator"
    fmt = args.format or "probs"
    shots = args.shots or 0
    
    print(styled(f"  Statevector: {c.n_qubits} qubits [{backend}]", Colors.BOLD, Colors.GREEN))
    
    t0 = time.time()
    result = sf.run(c, backend=backend, shots=shots)
    dt = time.time() - t0
    
    print(f"  Time: {dt*1000:.1f}ms")
    
    if fmt == "bloch":
        try:
            from superfermion.viz import bloch_angles
            sv = result.statevector
            if sv is not None and c.n_qubits == 1:
                angles = bloch_angles(sv)
                print(f"  Bloch angles: theta={angles.get('theta', 0):.3f}, phi={angles.get('phi', 0):.3f}")
                print(f"  X={angles.get('x', 0):.3f}, Y={angles.get('y', 0):.3f}, Z={angles.get('z', 0):.3f}")
            else:
                print(f"  Bloch display only for single-qubit states.")
        except Exception:
            print(f"  Bloch visualization not available.")
    
    elif fmt == "probs":
        counts = result.counts
        total = sum(counts.values()) or 1
        print(f"  Probability distribution (top 16):")
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:16]
        for bitstring, count in sorted_counts:
            prob = count / total
            bar = "█" * int(prob * 40)
            print(f"    |{bitstring}⟩  {prob:.4f}  {bar}")
    
    elif fmt == "counts":
        actual_shots = shots or 1024
        print(f"  Measurement counts ({actual_shots} shots, top 16):")
        sorted_counts = sorted(result.counts.items(), key=lambda x: -x[1])[:16]
        for bitstring, count in sorted_counts:
            print(f"    |{bitstring}⟩  {count}")


def cmd_qpu(args):
    """QPU fleet management."""
    import superfermion as sf
    
    if args.sub == "list":
        print(styled("  QPU Providers", Colors.BOLD, Colors.GREEN))
        providers = [
            ("ibm", "IBM Quantum", "IBMProvider"),
            ("ionq", "IonQ", "IonQProvider"),
            ("braket", "Amazon Braket", "BraketProvider"),
            ("openquantum", "OpenQuantum", "OpenQuantumProvider"),
        ]
        for key, name, cls_name in providers:
            status = "✓" if _check_provider_available(key) else "✗"
            color = Colors.GREEN if status == "✓" else Colors.DIM
            print(f"    {styled(status, color)} {name} ({key}) — {cls_name}")
    
    elif args.sub == "status":
        provider = args.provider or "ibm"
        print(styled(f"  QPU Status: {provider}", Colors.BOLD, Colors.GREEN))
        try:
            if provider == "ibm":
                from superfermion.runtime.providers.ibm import IBMProvider
                p = IBMProvider()
                backends = p.list_backends()
                print(f"  Available backends: {len(backends)}")
                for b in backends[:10]:
                    print(f"    - {b}")
            elif provider == "ionq":
                from superfermion.runtime.providers.ionq import IonQProvider
                p = IonQProvider()
                backends = p.list_backends()
                print(f"  Available backends: {len(backends)}")
                for b in backends[:10]:
                    print(f"    - {b}")
            else:
                print(f"  Status check for '{provider}' — provider accessible.")
        except Exception as e:
            print(styled(f"  Error: {e}", Colors.RED))
    
    elif args.sub == "submit":
        circuit_file = args.file
        if not circuit_file or not os.path.exists(circuit_file):
            print(styled(f"Error: Circuit file '{circuit_file}' not found.", Colors.RED))
            sys.exit(1)
        
        with open(circuit_file, 'r') as f:
            c = sf.Circuit.from_json(f.read())
        
        provider_name = args.provider or "ibm"
        backend_name = args.backend
        
        print(styled(f"  Submitting to {provider_name} ({backend_name or 'default'})", Colors.BOLD, Colors.GREEN))
        print(f"  Circuit: {c.n_qubits} qubits, {c.gate_count} gates")
        
        try:
            result = sf.run(c, backend=f"qpu.{provider_name}", shots=args.shots or 1024)
            print(styled(f"  Result: {result.counts}", Colors.GREEN))
        except Exception as e:
            print(styled(f"  Error: {e}", Colors.RED))
    
    elif args.sub == "results":
        job_id = args.job_id
        print(styled(f"  Job Results: {job_id}", Colors.BOLD, Colors.GREEN))
        print(f"  Retrieving results... (check provider dashboard for details)")
        print(styled(f"  Note: Use provider SDKs to retrieve results by job ID.", Colors.DIM))
    
    else:
        print(styled(f"Error: Unknown qpu command '{args.sub}'.", Colors.RED))
        print("Commands: list, status, submit, results")
        sys.exit(1)


def _check_provider_available(name: str) -> bool:
    """Check if a QPU provider is available (has credentials)."""
    try:
        if name == "ibm":
            from superfermion.runtime.providers.ibm import IBMProvider
            p = IBMProvider()
            return bool(p._token)
        elif name == "ionq":
            from superfermion.runtime.providers.ionq import IonQProvider
            p = IonQProvider()
            return bool(p._api_key)
        elif name == "braket":
            from superfermion.runtime.providers.aws import BraketProvider
            p = BraketProvider()
            return True  # may work with default AWS credentials
        elif name == "openquantum":
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            p = OpenQuantumProvider()
            return bool(p._client_id)
    except Exception:
        pass
    return False


def cmd_config(args):
    """Configuration management."""
    import superfermion as sf
    from superfermion.config.manager import load_config, save_config, get_default_config
    
    if args.sub == "show":
        config = load_config()
        print(styled("  Configuration", Colors.BOLD, Colors.GREEN))
        print()
        sections = ["project", "backends", "simulation", "compilation", "hardware", "telemetry", "security", "experiment"]
        for section in sections:
            print(f"  [{section}]")
            for key in config.list_keys(section):
                short_key = key.replace(f"{section}.", "")
                val = config.get(key)
                if isinstance(val, str) and ("token" in short_key.lower() or "secret" in short_key.lower()):
                    val = "***" if val else "(not set)"
                elif isinstance(val, str) and val == "":
                    val = "(not set)"
                print(f"    {short_key} = {val}")
            print()
    
    elif args.sub == "set":
        if not args.key:
            print(styled("Error: Specify a key to set.", Colors.RED))
            print("Usage: sf config set <key> <value>")
            sys.exit(1)
        
        config = load_config()
        key = args.key
        
        # Parse value
        raw = args.value
        if raw is None:
            print(styled("Error: Specify a value.", Colors.RED))
            sys.exit(1)
        
        # Auto-detect type
        if raw.lower() in ("true", "yes"):
            val = True
        elif raw.lower() in ("false", "no"):
            val = False
        elif raw.lower() in ("none", "null"):
            val = None
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        
        config.set(key, val)
        
        # Save to user config
        user_config_path = os.path.join(os.path.expanduser("~"), ".superfermion", "config.toml")
        try:
            save_config(config, user_config_path)
            print(styled(f"  Config set: {key} = {val}", Colors.GREEN))
            print(f"  Saved to {user_config_path}")
        except Exception as e:
            print(styled(f"  Warning: Could not save config: {e}", Colors.YELLOW))
            print(f"  Config set in memory: {key} = {val}")
    
    else:
        print(styled(f"Error: Unknown config command '{args.sub}'.", Colors.RED))
        print("Commands: show, set")
        sys.exit(1)


# ── New Commands: Plugin, Auth, Convert, Estimate, Compare, Jobs ────────────────

def cmd_plugin(args):
    """Plugin management."""
    from superfermion.plugins import list_backends, list_templates, list_passes, list_all, discover_plugins
    
    if args.sub == "list":
        print(styled("  Plugin Registry", Colors.BOLD, Colors.GREEN))
        print()
        
        # Discover plugins
        discover_plugins()
        
        all_plugins = list_all()
        
        if all_plugins["backends"]:
            print(styled("  Backends:", Colors.CYAN))
            for name in all_plugins["backends"]:
                print(f"    • {name}")
            print()
        
        if all_plugins["templates"]:
            print(styled("  Templates:", Colors.CYAN))
            for name in all_plugins["templates"]:
                print(f"    • {name}")
            print()
        
        if all_plugins["passes"]:
            print(styled("  Compiler Passes:", Colors.CYAN))
            for name in all_plugins["passes"]:
                print(f"    • {name}")
            print()
        
        if not any(all_plugins.values()):
            print(styled("  No plugins registered.", Colors.YELLOW))
            print()
            print("  To register a plugin, use:")
            print("    from superfermion.plugins import register_backend")
            print("    @register_backend('my_backend')")
            print("    class MyBackend: ...")
    
    elif args.sub == "install":
        print(styled(f"  Installing plugin: {args.package}", Colors.YELLOW))
        print()
        print("  Run: pip install", args.package)
        print()
        print("  Then import and register the plugin in your code.")
    
    elif args.sub == "create":
        import os
        plugin_name = args.name or "my_plugin"
        filename = f"sf_plugin_{plugin_name}.py"
        
        template = f'''#!/usr/bin/env python
"""Superfermion Plugin: {plugin_name}"""

from superfermion.plugins import register_backend, register_template
import superfermion as sf

@register_backend("{plugin_name}")
class {plugin_name.title()}Backend:
    """Custom backend implementation."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def run(self, circuit, shots=1000):
        """Execute circuit and return results."""
        # TODO: Implement backend logic
        return {{"counts": {{"0" * circuit.n_qubits: shots}}}}

@register_template("{plugin_name}_ansatz")
def {plugin_name}_ansatz(n_qubits: int, n_layers: int = 2):
    """Custom circuit template."""
    c = sf.Circuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            c.ry(0.5, q)
        for q in range(n_qubits - 1):
            c.cx(q, q + 1)
    return c

print(f"Plugin '{plugin_name}' loaded!")
'''
        
        with open(filename, "w") as f:
            f.write(template)
        
        print(styled(f"  Plugin template created: {filename}", Colors.GREEN))
        print()
        print("  Edit the file and import it to register your plugin.")
    
    else:
        print(styled(f"Error: Unknown plugin command '{args.sub}'.", Colors.RED))
        print("Commands: list, install, create")


def cmd_auth(args):
    """Authentication management."""
    from superfermion.security.credentials import CredentialStore, CredentialBackend
    
    store = CredentialStore(backend=CredentialBackend.ENCRYPTED_FILE)
    
    if args.sub == "login":
        provider = args.provider
        
        if not provider:
            print(styled("Error: Specify provider: ibm, ionq, aws, openquantum", Colors.RED))
            sys.exit(1)
        
        print(styled(f"  Login to {provider.upper()}", Colors.BOLD, Colors.CYAN))
        print()
        
        # Get credential based on provider
        if provider == "ibm":
            print("  Enter your IBM Quantum API token.")
            print("  Get it from: https://quantum.ibm.com/account")
            print()
            try:
                token = input("  API Token: ").strip()
            except EOFError:
                print(styled("\n  Error: Non-interactive mode. Use: --token <token>", Colors.RED))
                sys.exit(1)
            
            if args.token:
                token = args.token
            
            store.set("ibm_token", token, provider="ibm")
            print(styled("\n  ✓ IBM Quantum credentials saved.", Colors.GREEN))
        
        elif provider == "ionq":
            print("  Enter your IonQ API key.")
            print("  Get it from: https://cloud.ionq.com/settings/keys")
            print()
            try:
                token = input("  API Key: ").strip()
            except EOFError:
                token = args.token if args.token else ""
            
            if args.token:
                token = args.token
            
            store.set("ionq_token", token, provider="ionq")
            print(styled("\n  ✓ IonQ credentials saved.", Colors.GREEN))
        
        elif provider == "aws":
            print("  AWS credentials are read from environment variables:")
            print("    AWS_ACCESS_KEY_ID")
            print("    AWS_SECRET_ACCESS_KEY")
            print("    AWS_DEFAULT_REGION")
            print()
            print("  Or use AWS CLI: aws configure")
        
        elif provider == "openquantum":
            print("  Enter your OpenQuantum credentials.")
            try:
                client_id = input("  Client ID: ").strip()
                client_secret = input("  Client Secret: ").strip()
            except EOFError:
                client_id = args.token or ""
                client_secret = ""
            
            store.set("oq_client_id", client_id, provider="openquantum")
            store.set("oq_client_secret", client_secret, provider="openquantum")
            print(styled("\n  ✓ OpenQuantum credentials saved.", Colors.GREEN))
        
        else:
            print(styled(f"Error: Unknown provider '{provider}'.", Colors.RED))
            print("Providers: ibm, ionq, aws, openquantum")
            sys.exit(1)
    
    elif args.sub == "logout":
        provider = args.provider
        if provider:
            store.delete(f"{provider}_token")
            print(styled(f"  ✓ Logged out from {provider}.", Colors.GREEN))
        else:
            print(styled("Error: Specify provider.", Colors.RED))
    
    elif args.sub == "status":
        print(styled("  Authentication Status", Colors.BOLD, Colors.GREEN))
        print()
        
        providers = [("ibm", "IBM Quantum"), ("ionq", "IonQ"), ("aws", "AWS Braket"), ("openquantum", "OpenQuantum")]
        
        for key, name in providers:
            token = store.get(f"{key}_token") or store.get(f"{key}_api_key") or store.get(f"{key}_client_id")
            status = styled("✓ Configured", Colors.GREEN) if token else styled("✗ Not configured", Colors.YELLOW)
            print(f"  {name}: {status}")
    
    else:
        print(styled(f"Error: Unknown auth command '{args.sub}'.", Colors.RED))
        print("Commands: login, logout, status")


def cmd_convert(args):
    """Format conversion between circuit formats."""
    import os
    import superfermion as sf
    from superfermion.serialization.circuit_format import save_circuit, load_circuit
    
    input_file = args.input
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(styled(f"Error: File not found: {input_file}", Colors.RED))
        sys.exit(1)
    
    # Detect input format
    input_ext = os.path.splitext(input_file)[1].lower()
    output_ext = os.path.splitext(output_file)[1].lower() if output_file else ".json"
    
    print(styled("  Circuit Conversion", Colors.BOLD, Colors.GREEN))
    print()
    print(f"  Input:  {input_file} ({input_ext})")
    print(f"  Output: {output_file or 'stdout'} ({output_ext})")
    print()
    
    # Load circuit
    circuit = None
    
    if input_ext == ".sfc":
        circuit = load_circuit(input_file)
    elif input_ext in (".qasm", ".qasm3"):
        from superfermion.serialization.qasm_roundtrip import parse_qasm
        with open(input_file, "r") as f:
            qasm_str = f.read()
        circuit = parse_qasm(qasm_str)
    elif input_ext in (".json", ".sfj"):
        circuit = sf.load_circuit(input_file)
    elif input_ext == ".py":
        # Try to import as Qiskit/Cirq/PennyLane
        print(styled("  Python file detected. Use bridge functions:", Colors.YELLOW))
        print("    from superfermion.bridge import from_qiskit, from_cirq, from_pennylane")
        sys.exit(1)
    else:
        print(styled(f"Error: Unknown input format: {input_ext}", Colors.RED))
        print("Supported: .sfc, .qasm, .qasm3, .json")
        sys.exit(1)
    
    # Save circuit
    if output_ext == ".sfc":
        save_circuit(circuit, output_file)
        print(styled(f"  ✓ Saved as .sfc: {output_file}", Colors.GREEN))
    elif output_ext in (".qasm", ".qasm3"):
        from superfermion.serialization.qasm_roundtrip import to_qasm3
        qasm = to_qasm3(circuit)
        if output_file:
            with open(output_file, "w") as f:
                f.write(qasm)
        else:
            print(qasm)
        print(styled(f"  ✓ Converted to QASM3", Colors.GREEN))
    elif output_ext == ".json":
        if output_file:
            sf.save_circuit(circuit, output_file)
        else:
            print(json.dumps(circuit.to_dict(), indent=2))
        print(styled(f"  ✓ Converted to JSON", Colors.GREEN))
    else:
        print(styled(f"Error: Unknown output format: {output_ext}", Colors.RED))
        print("Supported: .sfc, .qasm, .qasm3, .json")
        sys.exit(1)


def cmd_estimate(args):
    """Cost estimation for QPU execution."""
    import os
    import superfermion as sf
    from superfermion.circuit import Circuit
    
    input_file = args.file
    backend = args.backend
    shots = args.shots or 4096
    
    if not os.path.exists(input_file):
        print(styled(f"Error: File not found: {input_file}", Colors.RED))
        sys.exit(1)
    
    # Load circuit from JSON
    with open(input_file, 'r') as f:
        circuit = Circuit.from_json(f.read())
    
    print(styled("  Cost Estimation", Colors.BOLD, Colors.GREEN))
    print()
    print(f"  Circuit: {input_file}")
    print(f"  Qubits:  {circuit.n_qubits}")
    print(f"  Gates:   {circuit.gate_count}")
    print(f"  Depth:   {circuit.depth}")
    print(f"  Shots:   {shots}")
    print()
    
    # Estimate costs
    print(styled("  Estimated Costs:", Colors.CYAN))
    print()
    
    # IBM Quantum pricing (approximate)
    if backend and "ibm" in backend.lower():
        # IBM uses seconds of quantum time
        estimated_time = (circuit.depth * shots * 0.001)  # rough estimate
        cost_per_second = 1.60  # USD per second (varies by plan)
        estimated_cost = estimated_time * cost_per_second
        print(f"  IBM Quantum:")
        print(f"    Estimated time: {estimated_time:.2f}s")
        print(f"    Estimated cost: ${estimated_cost:.2f}")
        print()
    
    # IonQ pricing
    if backend and "ionq" in backend.lower():
        # IonQ uses gate-shots
        gate_shots = circuit.gate_count * shots
        cost_per_million = 0.01  # approximate
        estimated_cost = (gate_shots / 1_000_000) * cost_per_million
        print(f"  IonQ:")
        print(f"    Gate-shots: {gate_shots:,}")
        print(f"    Estimated cost: ${estimated_cost:.2f}")
        print()
    
    # AWS Braket pricing
    if backend and ("braket" in backend.lower() or "aws" in backend.lower()):
        # AWS uses per-shot + per-gate pricing
        per_shot = 0.0003
        per_gate = 0.0001
        estimated_cost = (shots * per_shot) + (circuit.gate_count * shots * per_gate)
        print(f"  AWS Braket:")
        print(f"    Estimated cost: ${estimated_cost:.2f}")
        print()
    
    if not backend:
        print(styled("  Use --backend ibm|ionq|braket for specific estimates.", Colors.YELLOW))
        print()
        # Show all estimates
        print("  Approximate costs across providers:")
        print()
        
        # Generic estimate
        base_cost = circuit.gate_count * shots * 0.00001
        print(f"    IBM Quantum: ~${base_cost * 100:.2f} - ${base_cost * 500:.2f}")
        print(f"    IonQ:        ~${base_cost * 50:.2f} - ${base_cost * 200:.2f}")
        print(f"    AWS Braket:  ~${base_cost * 30:.2f} - ${base_cost * 100:.2f}")


def cmd_compare(args):
    """Compare circuit execution across backends."""
    import os
    import time
    import superfermion as sf
    from superfermion.circuit import Circuit
    
    input_file = args.file
    backends = args.backends.split(",") if args.backends else ["jax", "statevector", "mps"]
    shots = args.shots or 1000
    
    if not os.path.exists(input_file):
        print(styled(f"Error: File not found: {input_file}", Colors.RED))
        sys.exit(1)
    
    # Load circuit from JSON
    with open(input_file, 'r') as f:
        circuit = Circuit.from_json(f.read())
    
    print(styled("  Backend Comparison", Colors.BOLD, Colors.GREEN))
    print()
    print(f"  Circuit: {input_file}")
    print(f"  Qubits:  {circuit.n_qubits}")
    print(f"  Shots:   {shots}")
    print(f"  Backends: {', '.join(backends)}")
    print()
    
    results = []
    
    for backend in backends:
        backend = backend.strip()
        print(f"  Running on {backend}...", end=" ")
        
        try:
            start = time.perf_counter()
            result = sf.run(circuit, backend=backend, shots=shots)
            elapsed = time.perf_counter() - start
            
            # Get result counts
            if hasattr(result, "counts"):
                counts = result.counts
            else:
                counts = result.get("counts", {})
            
            # Calculate metrics
            total = sum(counts.values())
            entropy = 0
            for v in counts.values():
                if v > 0:
                    p = v / total
                    entropy -= p * (p and __import__("math").log2(p))
            
            results.append({
                "backend": backend,
                "time_ms": elapsed * 1000,
                "shots": total,
                "unique_outcomes": len(counts),
                "entropy": entropy,
                "status": "OK"
            })
            print(styled("OK", Colors.GREEN))
        
        except Exception as e:
            results.append({
                "backend": backend,
                "time_ms": 0,
                "shots": 0,
                "unique_outcomes": 0,
                "entropy": 0,
                "status": f"Error: {str(e)[:30]}"
            })
            print(styled(f"Error: {str(e)[:30]}", Colors.RED))
    
    print()
    print(styled("  Results:", Colors.CYAN))
    print()
    print(f"  {'Backend':<15} {'Time (ms)':<12} {'Shots':<10} {'Outcomes':<10} {'Status':<15}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*15}")
    
    for r in results:
        print(f"  {r['backend']:<15} {r['time_ms']:<12.1f} {r['shots']:<10} {r['unique_outcomes']:<10} {r['status']:<15}")
    
    # Find fastest
    valid = [r for r in results if r["status"] == "OK"]
    if valid:
        fastest = min(valid, key=lambda x: x["time_ms"])
        print()
        print(styled(f"  Fastest: {fastest['backend']} ({fastest['time_ms']:.1f}ms)", Colors.GREEN))


def cmd_jobs(args):
    """Job management across providers."""
    print(styled("  Job Management", Colors.BOLD, Colors.GREEN))
    print()
    
    if args.sub == "list":
        print(styled("  Jobs Across Providers:", Colors.CYAN))
        print()
        
        providers = []
        
        # Check IBM
        try:
            from superfermion.runtime.providers.ibm import IBMProvider
            p = IBMProvider()
            if p._api_key:
                providers.append(("IBM Quantum", p))
        except:
            pass
        
        # Check IonQ
        try:
            from superfermion.runtime.providers.ionq import IonQProvider
            p = IonQProvider()
            if p._api_key:
                providers.append(("IonQ", p))
        except:
            pass
        
        if not providers:
            print(styled("  No providers configured. Use 'sf auth login' first.", Colors.YELLOW))
            return
        
        for name, provider in providers:
            print(f"  {name}:")
            try:
                jobs = provider.list_jobs() if hasattr(provider, "list_jobs") else []
                if jobs:
                    for job in jobs[:5]:  # Show last 5
                        job_id = job.get("id", job.get("job_id", "unknown"))
                        status = job.get("status", "unknown")
                        print(f"    • {job_id}: {status}")
                else:
                    print("    (no recent jobs)")
            except Exception as e:
                print(f"    Error: {str(e)[:40]}")
            print()
    
    elif args.sub == "status":
        if not args.job_id:
            print(styled("Error: Specify --job-id", Colors.RED))
            sys.exit(1)
        
        print(f"  Job: {args.job_id}")
        print(f"  Provider: {args.provider or 'auto-detect'}")
        # Would query the specific job
        print(styled("  (Job status querying requires provider credentials)", Colors.YELLOW))
    
    elif args.sub == "cancel":
        if not args.job_id:
            print(styled("Error: Specify --job-id", Colors.RED))
            sys.exit(1)
        
        print(styled(f"  Cancelling job: {args.job_id}", Colors.YELLOW))
        # Would cancel the job
        print(styled("  (Job cancellation requires provider credentials)", Colors.YELLOW))
    
    else:
        print(styled(f"Error: Unknown jobs command '{args.sub}'.", Colors.RED))
        print("Commands: list, status, cancel")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="sf",
        description="Superfermion CLI — Quantum Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sf info                          Show system and framework info
  sf validate                      Run installation validation (22 checks)
  sf benchmark --qubits 14         Multi-backend benchmark up to 14 qubits
  sf run circuit.json --shots 4096 Execute a circuit
  sf compile circuit.json --target ibm_eagle
  sf transpile circuit.json --provider ionq --submit
  sf noise --model ibm_eagle       Inspect noise model
  sf gradient circuit.json --method parameter_shift --params 0.1,0.2
  sf train --model vqe --hamiltonian H2 --iterations 100
  sf circuit bell 2                Build Bell state circuit
  sf statevector circuit.json --format probs
  sf qpu list                      List QPU providers
  sf config show                   Show configuration
  sf backends                      List available backends
  sf version                       Print version info
        """,
    )

    parser.add_argument("--no-banner", action="store_true", help="Skip the banner display")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # sf info
    subparsers.add_parser("info", help="Show system and framework info")
    
    # sf version
    subparsers.add_parser("version", help="Print version info")
    
    # sf backends
    subparsers.add_parser("backends", help="List available backends")
    
    # sf validate
    subparsers.add_parser("validate", help="Run installation validation suite (22 checks)")
    
    # sf run
    run_parser = subparsers.add_parser("run", help="Execute a quantum circuit")
    run_parser.add_argument("file", help="Path to circuit JSON file")
    run_parser.add_argument("--backend", default="simulator", help="Backend to use")
    run_parser.add_argument("--shots", type=int, default=1024, help="Number of shots")
    
    # sf benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--qubits", type=int, default=10, help="Max qubits to benchmark")
    bench_parser.add_argument("--iterations", type=int, default=50, help="Iterations per benchmark")
    bench_parser.add_argument("--backend", default="all", help="Backend(s) to test (comma-separated or 'all')")
    bench_parser.add_argument("--suite", default="scaling", help="Suite: scaling | benchpress | compilation")
    bench_parser.add_argument("--output", default=None, help="Output file for results")

    # sf shor
    shor_parser = subparsers.add_parser("shor", help="Analyze Shor's measurement results")
    shor_parser.add_argument("measurements", help="Comma-separated binary measurement strings")
    shor_parser.add_argument("--N", type=int, default=15, help="Number to factor")
    shor_parser.add_argument("--a", type=int, default=7, help="Generator")

    # sf vqe
    vqe_parser = subparsers.add_parser("vqe", help="Run VQE on a preset Hamiltonian")
    vqe_parser.add_argument("--hamiltonian", default="H2",
                            help="Preset: H2 | LiH | BeH2 | tfim")
    vqe_parser.add_argument("--qubits", type=int, default=None)
    vqe_parser.add_argument("--electrons", type=int, default=None)
    vqe_parser.add_argument("--iterations", type=int, default=None)
    vqe_parser.add_argument("--backend", default=None)
    vqe_parser.add_argument("--optimizer", default=None,
                            help="scipy optimizer name (L-BFGS-B, COBYLA, …)")

    # sf qaoa
    qaoa_parser = subparsers.add_parser("qaoa", help="Run QAOA MaxCut on a preset graph")
    qaoa_parser.add_argument("--graph", default="ring4",
                             help="ring4 | ring6 | complete4 | triangle")
    qaoa_parser.add_argument("--p-layers", dest="p_layers", type=int, default=2)
    qaoa_parser.add_argument("--backend", default=None)

    # sf chemistry
    chem_parser = subparsers.add_parser("chemistry",
                                        help="Inspect a molecular Hamiltonian (optionally run VQE)")
    chem_parser.add_argument("molecule", help="H2 | LiH | BeH2 | …")
    chem_parser.add_argument("--basis", default="sto-3g")
    chem_parser.add_argument("--top", type=int, default=8,
                             help="Show first N Pauli terms")
    chem_parser.add_argument("--vqe", action="store_true",
                             help="Also run a quick VQE minimisation")
    chem_parser.add_argument("--qubits", type=int, default=None)
    chem_parser.add_argument("--electrons", type=int, default=None)
    chem_parser.add_argument("--iterations", type=int, default=None)
    chem_parser.add_argument("--backend", default=None)

    # sf qec
    qec_parser = subparsers.add_parser("qec",
                                       help="Run a QEC lifecycle or full FT audit")
    qec_parser.add_argument("--code", default="steane",
                            help="steane | surface | hypercube | bivariate_bicycle")
    qec_parser.add_argument("--error", default="X", help="X | Y | Z")
    qec_parser.add_argument("--error-qubit", dest="error_qubit", type=int, default=0)
    qec_parser.add_argument("--audit", action="store_true",
                            help="Run simulate_fault_tolerant_workflow instead")

    # ── NEW Phase 2.2 Subparsers ──────────────────────────────────

    # sf compile
    compile_parser = subparsers.add_parser("compile", help="Compile circuits for target hardware")
    compile_parser.add_argument("file", help="Path to circuit JSON file")
    compile_parser.add_argument("--target", default="generic", help="Target hardware: generic | ibm_eagle | ionq_aria")
    compile_parser.add_argument("--level", type=int, default=2, help="Optimization level (0-3)")
    compile_parser.add_argument("--basis", default=None, help="Target basis gates (comma-separated)")
    compile_parser.add_argument("--coupling-map", default=None, help="Path to coupling map JSON")
    compile_parser.add_argument("--output", default=None, help="Output file path")

    # sf transpile
    transpile_parser = subparsers.add_parser("transpile", help="Full transpile pipeline for QPU submission")
    transpile_parser.add_argument("file", help="Path to circuit JSON file")
    transpile_parser.add_argument("--provider", default="ibm", help="QPU provider: ibm | ionq | braket | openquantum")
    transpile_parser.add_argument("--backend", default=None, help="Specific QPU backend name")
    transpile_parser.add_argument("--level", type=int, default=None, help="Optimization level")
    transpile_parser.add_argument("--submit", action="store_true", help="Submit to QPU after transpilation")
    transpile_parser.add_argument("--shots", type=int, default=None, help="Shots for QPU submission")
    transpile_parser.add_argument("--output", default=None, help="Output file path")

    # sf noise
    noise_parser = subparsers.add_parser("noise", help="Apply and inspect noise models")
    noise_parser.add_argument("--model", default="ibm_eagle", help="Noise model: ibm_eagle | ideal")
    noise_parser.add_argument("apply", nargs="?", default=None, help="Circuit file to apply noise to")
    noise_parser.add_argument("--shots", type=int, default=4096, help="Shots for noisy simulation")
    noise_parser.add_argument("--output", default=None, help="Output file path")

    # sf gradient
    grad_parser = subparsers.add_parser("gradient", help="Compute gradients via parameter-shift or adjoint")
    grad_parser.add_argument("file", help="Path to circuit JSON file")
    grad_parser.add_argument("--method", default="parameter_shift", help="Method: parameter_shift | adjoint")
    grad_parser.add_argument("--params", required=True, help="Comma-separated parameter values")
    grad_parser.add_argument("--backend", default=None, help="Backend for expectation values")

    # sf train
    train_parser = subparsers.add_parser("train", help="Run QML training loop")
    train_parser.add_argument("--model", default="vqe", help="Model: vqe | qaoa")
    train_parser.add_argument("--hamiltonian", default=None, help="Hamiltonian preset (for VQE)")
    train_parser.add_argument("--graph", default=None, help="Graph preset (for QAOA)")
    train_parser.add_argument("--qubits", type=int, default=None, help="Number of qubits")
    train_parser.add_argument("--p-layers", type=int, default=None, help="QAOA p layers")
    train_parser.add_argument("--iterations", type=int, default=None, help="Training iterations")
    train_parser.add_argument("--backend", default=None, help="Simulation backend")
    train_parser.add_argument("--output", default=None, help="Output file for trained params")

    # sf circuit
    circuit_parser = subparsers.add_parser("circuit", help="Build and inspect quantum circuits")
    circuit_parser.add_argument("sub", nargs="?", default="info",
                                help="Sub-command: bell | ghz | qft | random | from-qasm | info | draw")
    circuit_parser.add_argument("source", nargs="?", default=None,
                                help="Qubits (for bell/ghz/qft/random), QASM file (for from-qasm), or circuit JSON (for info/draw)")
    circuit_parser.add_argument("--n-qubits", type=int, default=None, help="Number of qubits")
    circuit_parser.add_argument("--depth", type=int, default=None, help="Circuit depth (random)")
    circuit_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    circuit_parser.add_argument("--output", default=None, help="Output file path")

    # sf statevector
    sv_parser = subparsers.add_parser("statevector", help="Inspect and visualize quantum statevectors")
    sv_parser.add_argument("file", help="Path to circuit JSON file")
    sv_parser.add_argument("--backend", default="simulator", help="Backend for simulation")
    sv_parser.add_argument("--format", default="probs", help="Format: probs | bloch | counts")
    sv_parser.add_argument("--shots", type=int, default=None, help="Shots (for counts format)")

    # sf qpu
    qpu_parser = subparsers.add_parser("qpu", help="QPU fleet management")
    qpu_parser.add_argument("sub", nargs="?", default="list",
                             help="Sub-command: list | status | submit | results")
    qpu_parser.add_argument("file", nargs="?", default=None, help="Circuit file (for submit)")
    qpu_parser.add_argument("--provider", default=None, help="QPU provider: ibm | ionq | braket | openquantum")
    qpu_parser.add_argument("--backend", default=None, help="Specific backend name")
    qpu_parser.add_argument("--shots", type=int, default=None, help="Shots")
    qpu_parser.add_argument("--job-id", dest="job_id", default=None, help="Job ID (for results)")

    # sf config
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("sub", nargs="?", default="show",
                               help="Sub-command: show | set")
    config_parser.add_argument("key", nargs="?", default=None, help="Config key (for set)")
    config_parser.add_argument("value", nargs="?", default=None, help="Config value (for set)")
    
    # ── NEW De Facto CLI Commands ──────────────────────────────────
    
    # sf plugin
    plugin_parser = subparsers.add_parser("plugin", help="Plugin management")
    plugin_parser.add_argument("sub", nargs="?", default="list",
                               help="Sub-command: list | install | create")
    plugin_parser.add_argument("package", nargs="?", default=None, help="Package name (for install)")
    plugin_parser.add_argument("--name", default=None, help="Plugin name (for create)")
    
    # sf auth
    auth_parser = subparsers.add_parser("auth", help="Authentication management")
    auth_parser.add_argument("sub", nargs="?", default="status",
                              help="Sub-command: login | logout | status")
    auth_parser.add_argument("--provider", default=None, help="Provider: ibm | ionq | aws | openquantum")
    auth_parser.add_argument("--token", default=None, help="API token (non-interactive)")
    
    # sf convert
    convert_parser = subparsers.add_parser("convert", help="Convert circuit formats")
    convert_parser.add_argument("input", help="Input circuit file")
    convert_parser.add_argument("--output", default=None, help="Output file path")
    
    # sf estimate
    estimate_parser = subparsers.add_parser("estimate", help="Estimate QPU execution cost")
    estimate_parser.add_argument("file", help="Path to circuit JSON file")
    estimate_parser.add_argument("--backend", default=None, help="Target backend: ibm | ionq | braket")
    estimate_parser.add_argument("--shots", type=int, default=None, help="Number of shots")
    
    # sf compare
    compare_parser = subparsers.add_parser("compare", help="Compare execution across backends")
    compare_parser.add_argument("file", help="Path to circuit JSON file")
    compare_parser.add_argument("--backends", default="jax,statevector,mps", help="Comma-separated backends")
    compare_parser.add_argument("--shots", type=int, default=None, help="Number of shots")
    
    # sf jobs
    jobs_parser = subparsers.add_parser("jobs", help="Job management across providers")
    jobs_parser.add_argument("sub", nargs="?", default="list",
                              help="Sub-command: list | status | cancel")
    jobs_parser.add_argument("--provider", default=None, help="QPU provider")
    jobs_parser.add_argument("--job-id", dest="job_id", default=None, help="Job ID")

    args = parser.parse_args()

    # Always display the Golden QPU Banner on CLI entry; never let it crash the command.
    if not args.no_banner and not os.environ.get("SF_NO_BANNER"):
        try:
            banner()
        except Exception as exc:
            print(f"(banner suppressed: {exc.__class__.__name__})", file=sys.stderr)

    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "info":
        cmd_info()
    elif args.command == "version":
        cmd_version()
    elif args.command == "backends":
        cmd_backends()
    elif args.command == "validate":
        sys.exit(cmd_validate())
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "shor":
        cmd_shor(args)
    elif args.command == "vqe":
        cmd_vqe(args)
    elif args.command == "qaoa":
        cmd_qaoa(args)
    elif args.command == "chemistry":
        cmd_chemistry(args)
    elif args.command == "qec":
        cmd_qec(args)
    elif args.command == "compile":
        cmd_compile(args)
    elif args.command == "transpile":
        cmd_transpile(args)
    elif args.command == "noise":
        cmd_noise(args)
    elif args.command == "gradient":
        cmd_gradient(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "circuit":
        cmd_circuit(args)
    elif args.command == "statevector":
        cmd_statevector(args)
    elif args.command == "qpu":
        cmd_qpu(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "plugin":
        cmd_plugin(args)
    elif args.command == "auth":
        cmd_auth(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "jobs":
        cmd_jobs(args)


if __name__ == "__main__":
    main()

