"""
Variational algorithms: VQE and QAOA.

Uses scipy.optimize for classical optimization. Executes circuits via
``sf.run()`` — works with any device and simulation method.

Cross-validated against:
  Qiskit: VQE (qiskit_algorithms), QAOA (qiskit_algorithms)
  PennyLane: qml.AdamOptimizer + qml.expval

Usage:
    >>> from superfermion.algorithms.variational import VQE, QAOA
    >>> from superfermion.observables.core import SparsePauliOp
    >>> from superfermion.qml.templates import HardwareEfficientAnsatz

    # VQE on 2-qubit TFIM Hamiltonian
    >>> H = SparsePauliOp.from_dict({'ZZ': -1.0, 'XX': -0.5, 'II': 0.25})
    >>> ansatz = HardwareEfficientAnsatz(2, n_layers=2)
    >>> vqe = VQE(ansatz, H, device='cpu', method='statevector')
    >>> result = vqe.minimize()
    >>> print(result.optimal_value)  # ground state energy

    # QAOA on MaxCut (3-node triangle)
    >>> edges = [(0,1), (1,2), (0,2)]
    >>> qaoa = QAOA(3, edges, p_layers=2, device='cpu')
    >>> result = qaoa.minimize()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize

import superfermion as sf
from superfermion.algorithms.core import AlgorithmResult
from superfermion.observables.core import Observable, SparsePauliOp, PauliString, expval
from superfermion.qml.gradient.parameter_shift import parameter_shift_grad_vector
from superfermion.qml.gradient.adjoint import adjoint_grad_vector

logger = logging.getLogger(__name__)


# ── VQE ───────────────────────────────────────────────────────────────────────


class VQE:
    """Variational Quantum Eigensolver.

    Minimizes <psi(theta)|H|psi(theta)> using scipy.optimize.minimize.

    Args:
        ansatz:      Parametric SF Circuit (built with ``sf.param(...)`` gates).
        hamiltonian: SF observable (SparsePauliOp, Hamiltonian, PauliString).
        device:      Execution target — ``"cpu"``, ``"gpu"``, or a
                     ``DeviceExecutor`` object.
        method:      Simulation method — ``"statevector"``, ``"mps"``, etc.
        optimizer:   scipy optimizer method. Default 'L-BFGS-B' (uses gradients).
        shots:       0 = exact statevector (default); > 0 = shot-based sampling.
        diff_method: Gradient method: ``'adjoint'`` (default, Rust, O(M+N)*2^n)
                     or ``'parameter-shift'`` (2N forward passes).
    """

    def __init__(
        self,
        ansatz: sf.Circuit,
        hamiltonian: Observable,
        device: Any = "cpu",
        method: str = "statevector",
        optimizer: str = "L-BFGS-B",
        shots: int = 0,
        diff_method: str = "adjoint",
    ):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self._device = device
        self._method = method
        self.optimizer = optimizer
        self.shots = shots
        self.diff_method = diff_method
        self._param_names: List[str] = list(ansatz.parameters)

    def _energy(self, param_values: np.ndarray) -> float:
        params = {n: float(v) for n, v in zip(self._param_names, param_values)}
        bound = self.ansatz.bind(params)
        result = sf.run(bound, device=self._device, method=self._method, shots=self.shots)
        if result.statevector is not None:
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return float(np.real(self.hamiltonian._fast_expval(sv)))
        # Shot-based fallback
        from superfermion.qml.gradient.parameter_shift import _expval_from_counts
        return _expval_from_counts(result.counts, self.hamiltonian, self.ansatz.n_qubits)

    def _gradient(self, param_values: np.ndarray) -> np.ndarray:
        if self.diff_method == "adjoint" and self.shots == 0:
            return adjoint_grad_vector(
                self.ansatz,
                self.hamiltonian,
                self._param_names,
                param_values,
            )
        return parameter_shift_grad_vector(
            self.ansatz,
            self.hamiltonian,
            self._param_names,
            param_values,
            device=self._device,
            method=self._method,
            shots=self.shots,
        )

    def minimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        iterations: int = 1000,
        tol: float = 1e-8,
        seed: int = 42,
        callback_freq: int = 50,
        verbose: bool = False,
    ) -> AlgorithmResult:
        """Run VQE optimization.

        Args:
            initial_params: Starting parameter values. Random if None.
            iterations:     Max optimizer iterations.
            tol:            Convergence tolerance.
            seed:           Random seed for initial parameters.
            callback_freq:  Print energy every N iterations (if verbose=True).
            verbose:        Print optimization progress.

        Returns:
            AlgorithmResult with optimal_value, optimal_params, history.
        """
        n_params = len(self._param_names)
        if initial_params is None:
            rng = np.random.default_rng(seed)
            initial_params = rng.uniform(-math.pi, math.pi, n_params)

        history: List[float] = []

        def _cb(xk):
            e = self._energy(xk)
            history.append(e)
            if verbose and len(history) % callback_freq == 0:
                logger.info("  VQE iter %4d: E = %+.8f", len(history), e)

        use_jac = self.optimizer in ("L-BFGS-B", "CG", "BFGS", "trust-ncg")

        # Build optimizer options: only pass options each method actually accepts
        _opts: dict = {"maxiter": iterations}
        if self.optimizer == "L-BFGS-B":
            _opts["maxfun"] = iterations * 10  # L-BFGS-B specific
        elif self.optimizer == "COBYLA":
            _opts["rhobeg"] = 1.0  # COBYLA uses rhobeg, not maxiter directly

        scipy_result = scipy.optimize.minimize(
            fun=self._energy,
            x0=initial_params,
            jac=self._gradient if use_jac else None,
            method=self.optimizer,
            tol=tol,
            options=_opts,
            callback=_cb,
        )

        optimal_params = dict(zip(self._param_names, scipy_result.x.tolist()))

        return AlgorithmResult(
            optimal_value=float(scipy_result.fun),
            optimal_params=optimal_params,
            history=history,
            fidelity_history=None,
            final_fidelity=None,
            metadata={
                "scipy_success": scipy_result.success,
                "scipy_message": scipy_result.message,
                "n_fun_evals": scipy_result.nfev,
                "optimizer": self.optimizer,
                "device": str(self._device),
                "method": self._method,
                "n_params": n_params,
                "n_qubits": self.ansatz.n_qubits,
            },
        )


# ── QAOA ──────────────────────────────────────────────────────────────────────


class QAOA:
    """Quantum Approximate Optimization Algorithm.

    Implements p-layer QAOA for MaxCut and general Ising Hamiltonians.

    The cost Hamiltonian for a graph G = (V, E) with weights w_{ij}:
        H_C = Σ_{(i,j)∈E} w_{ij} * (I - Z_i Z_j) / 2

    The mixer Hamiltonian:
        H_B = Σ_i X_i

    QAOA circuit:  |s⟩ = H^n |0⟩
                   |γ,β⟩ = ∏_p e^{-iβ_p H_B} e^{-iγ_p H_C} |s⟩

    Args:
        n_qubits:         Number of qubits.
        edges:            List of (i, j) or (i, j, weight) tuples.
        p_layers:         QAOA depth.
        device:           Execution target — ``"cpu"``, ``"gpu"``, or
                          ``DeviceExecutor``.
        method:           Simulation method — ``"statevector"``, ``"mps"``, etc.
        optimizer:        Scipy method.
    """

    def __init__(
        self,
        n_qubits: int,
        edges: List[Tuple],
        p_layers: int = 1,
        device: Any = "cpu",
        method: str = "statevector",
        optimizer: str = "L-BFGS-B",
        shots: int = 0,
    ):
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self._device = device
        self._method = method
        self.optimizer = optimizer
        self.shots = shots

        # Parse edges: (i, j) or (i, j, w)
        self.edges: List[Tuple[int, int, float]] = []
        for e in edges:
            if len(e) == 2:
                self.edges.append((int(e[0]), int(e[1]), 1.0))
            else:
                self.edges.append((int(e[0]), int(e[1]), float(e[2])))

        # Build the cost Hamiltonian: H_C = Σ w_{ij}/2 * (I - Z_iZ_j)
        self.cost_hamiltonian = self._build_cost_hamiltonian()

    def _build_cost_hamiltonian(self) -> SparsePauliOp:
        """MaxCut cost: ½ Σ w_{ij} (I - Z_i Z_j)."""
        n = self.n_qubits
        terms: Dict[str, complex] = {}
        for i, j, w in self.edges:
            ii_str = 'I' * n
            zz_str = list('I' * n)
            zz_str[i] = 'Z'
            zz_str[j] = 'Z'
            zz_key = ''.join(zz_str)
            terms[ii_str] = terms.get(ii_str, 0) + w / 2
            terms[zz_key] = terms.get(zz_key, 0) - w / 2
        return SparsePauliOp.from_dict(terms)

    def _build_circuit(self, gamma: np.ndarray, beta: np.ndarray) -> sf.Circuit:
        """Build the QAOA circuit with concrete angle values."""
        n = self.n_qubits
        c = sf.Circuit(n)

        # Initial state: uniform superposition
        for i in range(n):
            c.h(i)

        for p in range(self.p_layers):
            g = float(gamma[p])
            b = float(beta[p])

            # Cost layer: e^{-i γ H_C}
            # For each ZZ term: e^{-i γ w/2 * (-ZZ)} = CNOT RZ(2γw) CNOT
            for qi, qj, w in self.edges:
                c.cx(qi, qj)
                c.rz(2.0 * g * w, qj)
                c.cx(qi, qj)

            # Mixer layer: e^{-i β H_B} = Π_i RX(2β)
            for i in range(n):
                c.rx(2.0 * b, i)

        return c

    def _cost(self, params: np.ndarray) -> float:
        """Return NEGATIVE ⟨H_C⟩ so scipy.minimize maximizes the cut."""
        gamma = params[:self.p_layers]
        beta  = params[self.p_layers:]
        c = self._build_circuit(gamma, beta)
        result = sf.run(c, device=self._device, method=self._method, shots=self.shots)

        if result.statevector is not None:
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return -float(np.real(self.cost_hamiltonian._fast_expval(sv)))
        from superfermion.qml.gradient.parameter_shift import _expval_from_counts
        return -_expval_from_counts(result.counts, self.cost_hamiltonian, self.n_qubits)

    def minimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        iterations: int = 1000,
        seed: int = 42,
        verbose: bool = False,
    ) -> AlgorithmResult:
        """Run QAOA optimization.

        Returns:
            AlgorithmResult with:
              - optimal_value: minimum ⟨H_C⟩ (negated MaxCut value)
              - optimal_params: {'gamma': [...], 'beta': [...]}
              - history: energy trace
              - metadata['max_cut_value']: best integer cut value
              - metadata['best_bitstring']: best measured bitstring
        """
        if initial_params is None:
            rng = np.random.default_rng(seed)
            initial_params = np.concatenate([
                rng.uniform(0, math.pi, self.p_layers),
                rng.uniform(0, math.pi / 2, self.p_layers),
            ])

        history: List[float] = []

        def _cb(xk):
            history.append(self._cost(xk))
            if verbose and len(history) % 20 == 0:
                logger.info("  QAOA iter %4d: ⟨H_C⟩ = %+.6f", len(history), history[-1])

        res = scipy.optimize.minimize(
            fun=self._cost,
            x0=initial_params,
            method=self.optimizer,
            tol=1e-8,
            options={"maxiter": iterations},
            callback=_cb,
        )

        opt_gamma = res.x[:self.p_layers].tolist()
        opt_beta  = res.x[self.p_layers:].tolist()

        # Sample the optimized circuit to get the best bitstring
        opt_circuit = self._build_circuit(np.array(opt_gamma), np.array(opt_beta))
        sample_result = sf.run(opt_circuit, device=self._device, method=self._method, shots=8192)
        best_bitstring = max(sample_result.counts, key=lambda b: sample_result.counts[b]) if sample_result.counts else "0" * self.n_qubits
        max_cut = self._cut_value(best_bitstring)

        return AlgorithmResult(
            optimal_value=-float(res.fun),  # positive: max ⟨H_C⟩
            optimal_params={"gamma": opt_gamma, "beta": opt_beta},
            history=[-e for e in history],  # convert to positive expvals
            fidelity_history=None,
            final_fidelity=None,
            metadata={
                "max_cut_value": max_cut,
                "best_bitstring": best_bitstring,
                "scipy_success": res.success,
                "optimizer": self.optimizer,
                "p_layers": self.p_layers,
                "device": str(self._device),
                "method": self._method,
                "n_qubits": self.n_qubits,
            },
        )

    def _cut_value(self, bitstring: str) -> float:
        """Compute cut value for a given bitstring assignment."""
        assignment = [int(b) for b in bitstring]
        cut = 0.0
        for i, j, w in self.edges:
            if assignment[i] != assignment[j]:
                cut += float(w)  # float weights, not int — int(0.85)=0 is wrong
        return cut
