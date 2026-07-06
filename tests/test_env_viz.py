"""
Test Environment + Visualization modules (Phase 4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.environment import detect_environment, get_display_backend, display_circuit_html
from superfermion.viz import bloch_angles, state_bar_chart, convergence_plot_ascii


def test_environment_detection():
    print("Testing environment detection...")
    env = detect_environment()
    backend = get_display_backend()
    print(f"  Detected: {env.name}")
    print(f"  Display backend: {backend}")
    assert env is not None
    assert backend in ("html", "svg", "text")
    print("[PASS] Environment detection verified.")


def test_circuit_html():
    print("\nTesting circuit HTML rendering...")
    c = sf.Circuit(2)
    c.h(0).cx(0, 1)
    html = display_circuit_html(c)
    assert "Qubits: 2" in html
    assert "Gates:" in html
    print(f"  HTML length: {len(html)} chars")
    print("[PASS] Circuit HTML rendering verified.")


def test_bloch_angles():
    print("\nTesting Bloch sphere angles...")
    # |0> state: should be north pole (theta=0)
    sv_0 = jnp.array([1, 0], dtype=jnp.complex64)
    angles = bloch_angles(sv_0)
    print(f"  |0> angles: theta={angles['theta']:.4f}, z={angles['z']:.4f}")
    assert abs(angles['z'] - 1.0) < 0.01
    
    # |1> state: should be south pole (theta=pi)
    sv_1 = jnp.array([0, 1], dtype=jnp.complex64)
    angles = bloch_angles(sv_1)
    print(f"  |1> angles: theta={angles['theta']:.4f}, z={angles['z']:.4f}")
    assert abs(angles['z'] - (-1.0)) < 0.01
    
    # |+> state: should be on equator (x=1)
    sv_plus = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)], dtype=jnp.complex64)
    angles = bloch_angles(sv_plus)
    print(f"  |+> angles: theta={angles['theta']:.4f}, x={angles['x']:.4f}")
    assert abs(angles['x'] - 1.0) < 0.1
    
    print("[PASS] Bloch angles verified.")


def test_state_bar_chart():
    print("\nTesting state probability bar chart...")
    # Bell state
    sv = jnp.array([1/jnp.sqrt(2), 0, 0, 1/jnp.sqrt(2)], dtype=jnp.complex64)
    chart = state_bar_chart(sv, n_qubits=2)
    print(chart)
    assert "|00>" in chart
    assert "|11>" in chart
    print("[PASS] State bar chart verified.")


def test_convergence_plot():
    print("\nTesting convergence plot...")
    import math
    history = [1.0 - 0.9 * (1 - math.exp(-i/20)) for i in range(100)]
    plot = convergence_plot_ascii(history, width=40, height=8)
    print(plot)
    assert "iter=100" in plot
    print("[PASS] Convergence plot verified.")


if __name__ == "__main__":
    try:
        test_environment_detection()
        test_circuit_html()
        test_bloch_angles()
        test_state_bar_chart()
        test_convergence_plot()
        print("\nSession 24: Environment + Visualization ready.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
