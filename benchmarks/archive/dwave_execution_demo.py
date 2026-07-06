"""
Superfermion D-Wave Execution Demo.
Shows how a gate-based QAOA-style circuit is automatically translated 
and executed on D-Wave Advantage hardware.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import superfermion as sf
from superfermion.circuit import Circuit
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_dwave_demo():
    console.print(Panel.fit(
        "[bold gold1]D-Wave Hardware Execution via Superfermion[/bold gold1]\n"
        "[dim]Translating Gate-Based Circuit to Quantum Annealing QUBO[/dim]",
        border_style="gold1"
    ))

    # 1. Define a 4-qubit Max-Cut problem (Ising style)
    # We want to find a cut for a square graph 0-1-2-3-0
    c = Circuit(4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    # In Superfermion, we represent the problem as a circuit with RZZ gates
    # This matches the QAOA cost Hamiltonian layer logic.
    gamma = 1.0
    for u, v in edges:
        c.rzz(gamma, u, v)  # Edge (u,v)
        
    console.print(f"\n[bold cyan]1. Defined Circuit for Max-Cut Problem:[/bold cyan]")
    console.print(f"   Qubits: {c.n_qubits}")
    console.print(f"   Gates:  {c.gate_count} (RZZ gates for edges)")
    
    # 2. Run on D-Wave backend
    # Superfermion automatically recognizes 'dwave' and uses the Ising bridge.
    console.print(f"\n[bold green]2. Dispatching to D-Wave Advantage QPU...[/bold green]")
    
    # We pass target='dwave_advantage' to trigger spec-based compilation
    # and backend='dwave' to use the annealer bridge.
    result = sf.run(c, backend="dwave", target="dwave_advantage", shots=2000)
    
    console.print(f"\n[bold magenta]3. D-Wave Execution Results:[/bold magenta]")
    console.print(f"   Solver: {result.metadata['solver']}")
    console.print(f"   QPU Time: {result.metadata['qpu_access_time_us']} μs")
    console.print(f"   Top Samples (States):")
    
    for state, count in sorted(result.counts.items(), key=lambda x: -x[1])[:3]:
        # Map 1/0 to sets A/B for Max-Cut
        set_a = [i for i, b in enumerate(state) if b == '1']
        set_b = [i for i, b in enumerate(state) if b == '0']
        console.print(f"     [bold]{state}[/bold]: {count} shots | Set A: {set_a}, Set B: {set_b}")

    console.print("\n[bold gold1]MISSION SUCCESS[/bold gold1]")
    console.print("[dim]Superfermion: Bridge between Gate-Based and Annealing-Based Quantum Computing.[/dim]")

if __name__ == "__main__":
    run_dwave_demo()
