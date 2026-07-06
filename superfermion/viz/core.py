"""
Visualization — Circuit diagrams, Bloch sphere, state visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import jax.numpy as jnp


def bloch_angles(statevector: jnp.ndarray, qubit: int = 0) -> Dict[str, float]:
    """Extract Bloch sphere angles (theta, phi) for a single qubit.
    
    For multi-qubit states, traces out all other qubits first.
    
    Args:
        statevector: The quantum state as a JAX array.
        qubit: Which qubit to extract (0-indexed).
        
    Returns:
        Dict with keys 'theta', 'phi', 'x', 'y', 'z' (Bloch coordinates).
    """
    n_qubits = int(math.log2(len(statevector)))
    
    if n_qubits == 1:
        alpha, beta = complex(statevector[0]), complex(statevector[1])
    else:
        # Partial trace: get the reduced density matrix for this qubit
        dim = 2**n_qubits
        rho = jnp.outer(statevector, jnp.conj(statevector))
        
        # Trace out all qubits except the target
        rho_reduced = _partial_trace(rho, qubit, n_qubits)
        
        # Extract Bloch vector from density matrix
        # rho = (I + r.sigma) / 2
        x = float(jnp.real(rho_reduced[0, 1] + rho_reduced[1, 0]))
        y = float(jnp.real(1j * (rho_reduced[0, 1] - rho_reduced[1, 0])))
        z = float(jnp.real(rho_reduced[0, 0] - rho_reduced[1, 1]))
        
        theta = math.acos(max(-1, min(1, z)))
        phi = math.atan2(y, x)
        
        return {"theta": theta, "phi": phi, "x": x, "y": y, "z": z}
    
    # Single qubit case
    r = abs(alpha)**2 + abs(beta)**2
    if r < 1e-10:
        return {"theta": 0, "phi": 0, "x": 0, "y": 0, "z": 1}
    
    theta = 2 * math.acos(min(1.0, abs(alpha) / math.sqrt(r)))
    if abs(beta) > 1e-10:
        phi = float(jnp.angle(beta) - jnp.angle(alpha))
    else:
        phi = 0.0
    
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    
    return {"theta": theta, "phi": phi, "x": x, "y": y, "z": z}


def _partial_trace(rho: jnp.ndarray, keep_qubit: int, n_qubits: int) -> jnp.ndarray:
    """Compute the partial trace, keeping only one qubit."""
    dim = 2**n_qubits
    rho_reshaped = rho.reshape([2]*n_qubits*2)
    
    # Build the axes to trace over
    trace_axes = []
    for i in range(n_qubits):
        if i != keep_qubit:
            trace_axes.append(i)
    
    # Trace out other qubits (pair each bra with its ket)
    result = rho_reshaped
    offset = 0
    for ax in sorted(trace_axes, reverse=True):
        result = jnp.trace(result, axis1=ax - offset, axis2=ax + n_qubits - 2*offset - offset)
        offset += 1
    
    return result.reshape(2, 2)


def state_bar_chart(statevector: jnp.ndarray, n_qubits: int = None) -> str:
    """Generate an ASCII bar chart of state probabilities.
    
    Args:
        statevector: The quantum state.
        n_qubits: Number of qubits (auto-detected if None).
        
    Returns:
        ASCII string with probability bars.
    """
    probs = jnp.abs(statevector)**2
    if n_qubits is None:
        n_qubits = int(math.log2(len(probs)))
    
    lines = []
    max_bar = 40
    
    for i, p in enumerate(probs):
        p_val = float(p)
        if p_val < 1e-6:
            continue
        basis = format(i, f'0{n_qubits}b')
        bar_len = int(p_val * max_bar)
        bar = '#' * bar_len + '-' * (max_bar - bar_len)
        lines.append(f"  |{basis}> [{bar}] {p_val:.4f}")
    
    if not lines:
        lines.append("  (all zero)")
    
    return "\n".join(lines)


def convergence_plot_ascii(history: List[float], width: int = 60, height: int = 15) -> str:
    """Generate an ASCII convergence plot from a loss history.
    
    Args:
        history: List of loss values over iterations.
        width: Plot width in characters.
        height: Plot height in lines.
        
    Returns:
        ASCII string of the convergence curve.
    """
    if not history:
        return "  (no data)"
    
    mn, mx = min(history), max(history)
    rng = mx - mn if mx != mn else 1.0
    
    # Downsample if needed
    step = max(1, len(history) // width)
    sampled = history[::step][:width]
    
    lines = []
    for row in range(height):
        threshold = mx - (row / (height - 1)) * rng
        line = ""
        for val in sampled:
            if val <= threshold:
                line += "*"
            else:
                line += " "
        
        label = f"{threshold:>10.4f}"
        lines.append(f"  {label} |{line}")
    
    # X-axis
    lines.append(f"  {'':>10} +{''.join(['-'] * len(sampled))}")
    lines.append(f"  {'':>10}  0{'':>{len(sampled)-2}}iter={len(history)}")
    
    return "\n".join(lines)


# ── Matplotlib visualizations (optional dependency) ──────────────────────────

def draw_mpl(circuit, figsize=None, style: str = "qiskit"):
    """Draw a circuit diagram using matplotlib (Qiskit-style).

    Requires: ``pip install matplotlib``

    Args:
        circuit: An ``sf.Circuit`` instance.
        figsize: Tuple (width, height) in inches. Auto-scaled if None.
        style:  Drawing style: 'qiskit' (default) or 'textbook'.

    Returns:
        ``matplotlib.figure.Figure``

    Example:
        >>> import superfermion as sf
        >>> from superfermion.viz import draw_mpl
        >>> c = sf.Circuit(3); c.h(0).cx(0, 1).cx(1, 2)
        >>> fig = draw_mpl(c)
        >>> fig.savefig("circuit.png")
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for draw_mpl(). Install with: pip install matplotlib"
        )

    gates = list(circuit._gates) if hasattr(circuit, "_gates") else []
    n_qubits = circuit.n_qubits

    if not gates:
        if figsize is None:
            figsize = (4, n_qubits * 0.6 + 0.5)
        fig, ax = plt.subplots(figsize=figsize)
        for q in range(n_qubits):
            ax.plot([0, 1], [q, q], "k-", lw=1)
            ax.text(-0.3, q, f"q{q}", ha="right", va="center", fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.6, n_qubits - 0.4)
        ax.axis("off")
        ax.set_title("Circuit (empty)")
        return fig

    # Compute gate positions from depth
    qubit_times = {q: 0 for q in range(n_qubits)}
    gate_positions = []
    for g in gates:
        qbs = list(g.qubits)
        t0 = max(qubit_times.get(q, 0) for q in qbs)
        gate_positions.append((t0, g))
        for q in qbs:
            qubit_times[q] = t0 + 1

    max_time = max(qubit_times.values()) if qubit_times else 0

    if figsize is None:
        figsize = (max(4, max_time * 0.8), max(2, n_qubits * 0.6))
    fig, ax = plt.subplots(figsize=figsize)

    # Draw wires
    for q in range(n_qubits):
        ax.plot([-0.3, max_time + 0.3], [q, q], "k-", lw=1, alpha=0.4)
        ax.text(-0.5, q, f"$q_{q}$", ha="right", va="center", fontsize=11)

    # Draw gates
    gate_colors = {
        "H": "#FFB347", "X": "#FF6B6B", "Y": "#77DD77", "Z": "#AEC6CF",
        "RX": "#FFB6C1", "RY": "#FFDAB9", "RZ": "#B0E0E6",
        "S": "#DDA0DD", "T": "#F0E68C", "P": "#E6E6FA",
        "U": "#D8BFD8",
        "CX": "#87CEEB", "CNOT": "#87CEEB", "CZ": "#98FB98",
        "CY": "#98FB98", "SWAP": "#FFD700",
        "TOFFOLI": "#FFA07A", "CCX": "#FFA07A",
        "MEASURE": "#808080", "BARRIER": "#D3D3D3",
    }

    for t0, g in gate_positions:
        name = g.name.upper()
        qbs = list(g.qubits)
        color = gate_colors.get(name, "#FFFFFF")

        if name in ("CX", "CNOT", "CY", "CZ"):
            ctrl, tgt = qbs[0], qbs[1]
            ax.plot(t0, ctrl, "ko", markersize=5)
            ax.plot([t0, t0], [ctrl, tgt], "k-", lw=1.2)
            tgt_marker = "\u2295" if name in ("CX", "CNOT") else "\u25CB"
            circle = plt.Circle((t0, tgt), 0.18, fc="white", ec="k", lw=1.2)
            ax.add_patch(circle)
            ax.text(t0, tgt, tgt_marker, ha="center", va="center", fontsize=10)
        elif name == "SWAP":
            ax.plot([t0, t0], [qbs[0], qbs[1]], "k-", lw=1.2)
            ax.plot(t0, qbs[0], "kx", markersize=6)
            ax.plot(t0, qbs[1], "kx", markersize=6)
        elif name == "TOFFOLI" or name == "CCX":
            c1, c2, tgt = qbs[0], qbs[1], qbs[2]
            ax.plot(t0, c1, "ko", markersize=4)
            ax.plot(t0, c2, "ko", markersize=4)
            ax.plot([t0, t0], [min(qbs), max(qbs)], "k-", lw=1.2)
            circle = plt.Circle((t0, tgt), 0.18, fc="white", ec="k", lw=1.2)
            ax.add_patch(circle)
            ax.text(t0, tgt, "\u2295", ha="center", va="center", fontsize=10)
        elif name == "MEASURE":
            rect = mpatches.Rectangle(
                (t0 - 0.25, qbs[0] - 0.2), 0.5, 0.4,
                fc="#D3D3D3", ec="k", lw=1
            )
            ax.add_patch(rect)
            ax.text(t0, qbs[0], "M", ha="center", va="center", fontsize=8)
        elif name == "BARRIER":
            ax.plot([t0, t0], [-0.3, n_qubits - 0.7], "k--", lw=0.8, alpha=0.5)
        else:
            # Single-qubit gate box
            label = name
            if g.params and len(g.params) > 0:
                p = g.params[0]
                if isinstance(p, (int, float)):
                    label = f"{name}({p:.2g})"
            rect = mpatches.FancyBboxPatch(
                (t0 - 0.28, qbs[0] - 0.22), 0.56, 0.44,
                boxstyle="round,pad=0.02", fc=color, ec="k", lw=1.2
            )
            ax.add_patch(rect)
            ax.text(t0, qbs[0], label, ha="center", va="center", fontsize=7, weight="bold")

    ax.set_xlim(-0.8, max_time + 0.5)
    ax.set_ylim(-0.7, n_qubits - 0.3)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"Circuit ({n_qubits} qubits, {len(gates)} gates, depth={circuit.depth})")

    return fig


def plot_histogram(counts: dict, title: str = "Measurement Results",
                   figsize=None, sort: str = "desc"):
    """Plot a measurement outcome histogram.

    Requires: ``pip install matplotlib``

    Args:
        counts:  Dictionary mapping bitstrings to counts (e.g., ``{'00': 512, '11': 512}``).
        title:   Plot title.
        figsize: Figure size (width, height) in inches.
        sort:    Sort order: 'desc' (most frequent first), 'asc', or 'none'.

    Returns:
        ``matplotlib.figure.Figure``

    Example:
        >>> from superfermion.viz import plot_histogram
        >>> fig = plot_histogram({'00': 500, '01': 200, '10': 180, '11': 520})
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_histogram(). Install with: pip install matplotlib"
        )

    items = list(counts.items())
    if sort == "desc":
        items.sort(key=lambda x: -x[1])
    elif sort == "asc":
        items.sort(key=lambda x: x[1])

    labels = [k for k, _ in items]
    values = [v for k, v in items]
    total = sum(values)

    if figsize is None:
        figsize = (max(4, len(labels) * 0.8), 4)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis([v / max(values) * 0.8 for v in values])
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="k", lw=0.5)

    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"|{l}>" for l in labels], rotation=45, ha="right")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    return fig


def plot_bloch(statevector, qubit: int = 0, figsize=(5, 5),
               title: str = "Bloch Sphere"):
    """Render a Bloch sphere for a single qubit in 3D.

    Requires: ``pip install matplotlib``

    Args:
        statevector: The quantum state (JAX or numpy array).
        qubit:       Which qubit to visualize (0-indexed). Partial trace for multi-qubit.
        figsize:     Figure size in inches.
        title:       Plot title.

    Returns:
        ``matplotlib.figure.Figure``

    Example:
        >>> from superfermion.viz import plot_bloch
        >>> import jax.numpy as jnp
        >>> sv = jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)  # |+> state
        >>> fig = plot_bloch(sv)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_bloch(). Install with: pip install matplotlib"
        )

    angles = bloch_angles(statevector, qubit)
    x, y, z = angles["x"], angles["y"], angles["z"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="lightgray", alpha=0.3, lw=0.3)

    # Draw axes
    for axis, color, label in [
        ([0, 0, 1.3], "red", "|0>"),
        ([0, 0, -1.3], "red", "|1>"),
        ([0, 1.3, 0], "green", "|+i>"),
        ([0, -1.3, 0], "green", "|-i>"),
        ([1.3, 0, 0], "blue", "|+>"),
        ([-1.3, 0, 0], "blue", "|->"),
    ]:
        ax.quiver(0, 0, 0, *axis, color=color, alpha=0.3, lw=0.5,
                  arrow_length_ratio=0.05)

    # State vector arrow
    ax.quiver(0, 0, 0, x, y, z, color="purple", lw=2.5, arrow_length_ratio=0.1)

    # Blip at tip
    ax.scatter([x], [y], [z], color="purple", s=60)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\n(|{'>' if abs(x - 1) < 0.01 else ('0>' if abs(z - 1) < 0.01 else '+')}>)")
    return fig


def plot_state_city(density_matrix, figsize=(8, 6), title: str = "State City Plot"):
    """Plot a cityscape visualization of a density matrix.

    Real parts shown as upward bars, imaginary as colored surface.

    Requires: ``pip install matplotlib``

    Args:
        density_matrix: 2-D complex array representing the density matrix.
        figsize:        Figure size in inches.
        title:          Plot title.

    Returns:
        ``matplotlib.figure.Figure``
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_state_city(). Install with: pip install matplotlib"
        )

    dm = np.asarray(density_matrix)
    real_part = np.real(dm)
    imag_part = np.imag(dm)
    n = dm.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(range(n), range(n))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)

    dx = dy = 0.6
    dz = real_part.flatten()

    # Color bars by imaginary component
    colors = plt.cm.RdBu((imag_part.flatten() + 1) / 2)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, edgecolor="k", lw=0.2, alpha=0.85)
    ax.set_xlabel("Row")
    ax.set_ylabel("Column")
    ax.set_zlabel("Re(\u03c1)")
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    return fig


def draw_latex(circuit, document: bool = False) -> str:
    """Export a circuit to LaTeX using the quantikz package.

    Args:
        circuit:  An ``sf.Circuit`` instance.
        document: If True, wrap in a full LaTeX document. If False, return only the tikz code.

    Returns:
        LaTeX source string.

    Example:
        >>> from superfermion.viz import draw_latex
        >>> c = sf.Circuit(2); c.h(0).cx(0, 1)
        >>> print(draw_latex(c))
    """
    gates = list(circuit._gates) if hasattr(circuit, "_gates") else []
    n = circuit.n_qubits

    # Map gate to quantikz command
    gate_map = {
        "H": "\\gate{{H}}",
        "X": "\\gate{{X}}",
        "Y": "\\gate{{Y}}",
        "Z": "\\gate{{Z}}",
        "S": "\\gate{{S}}",
        "T": "\\gate{{T}}",
        "RX": "\\gate{{R_x}}",
        "RY": "\\gate{{R_y}}",
        "RZ": "\\gate{{R_z}}",
        "P": "\\gate{{P}}",
    }

    # Assign gates to columns per qubit
    qubit_cols = {q: 0 for q in range(n)}
    columns = [[] for _ in range(n)]

    def _get_qubits(g):
        """Extract actual qubit indices from a gate, handling param-qubit swap."""
        qbs = list(g.qubits)
        if qbs and all(isinstance(q, (int,)) for q in qbs):
            return qbs
        # For parameterized single-qubit gates, qubits may contain the angle;
        # actual qubit indices are in params.
        if g.params and all(isinstance(p, (int,)) for p in g.params):
            return list(g.params)
        return [q for q in qbs if isinstance(q, int)]

    for g in gates:
        qbs = _get_qubits(g)
        if not qbs:
            continue
        col = max(qubit_cols[q] for q in qbs)
        name = g.name.upper()

        if name in ("CX", "CNOT"):
            columns[qbs[0]].append((col, "\\ctrl{1}"))
            columns[qbs[1]].append((col, "\\targ{}"))
        elif name == "CZ":
            columns[qbs[0]].append((col, "\\ctrl{1}"))
            columns[qbs[1]].append((col, "\\ctrl{1}"))
        elif name == "SWAP":
            columns[qbs[0]].append((col, "\\swap{1}"))
            columns[qbs[1]].append((col, "\\targX{}"))
        else:
            cmd = gate_map.get(name, f"\\gate{{{name}}}")
            columns[qbs[0]].append((col, cmd))

        for q in qbs:
            qubit_cols[q] = col + 1

    max_col = max(qubit_cols.values()) if qubit_cols else 0

    # Build rows
    lines = []
    for q in range(n):
        row_parts = ["\\qw"] * (max_col + 1)
        for col, cmd in columns[q]:
            row_parts[col] = cmd
        row_parts.append("\\qw")
        lines.append(" & ".join(row_parts))

    body = " \\\n".join(lines) + " \\\n"
    tikz = (
        f"\\begin{{quantikz}}\n"
        f"{body}"
        f"\\end{{quantikz}}"
    )

    if document:
        return (
            f"\\documentclass{{standalone}}\n"
            f"\\usepackage{{quantikz}}\n"
            f"\\begin{{document}}\n"
            f"{tikz}\n"
            f"\\end{{document}}\n"
        )
    return tikz
