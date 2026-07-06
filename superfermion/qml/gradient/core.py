"""
JAX Integration Core — Custom primitives for differentiable quantum circuits.

This module defines the `quantum_circuit_p` JAX primitive, which allows
Superfermion circuits to be treated as differentiable JAX functions.

Key Features:
    - JVP (Jacobian-Vector Product): Implemented via the parameter-shift rule.
    - VMap (Vectorized Map): Batching support for quantum circuits.
    - JIT (Just-In-Time) compilation: Traceable through JAX.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import ad, batching, mlir

from superfermion.circuit import Circuit
from superfermion.runner import run


# ───────────────────────────────────────────────────────────
# 1. Define the JAX Primitive
# ───────────────────────────────────────────────────────────

if hasattr(core, "Primitive"):
    Primitive = core.Primitive
else:
    # Handle older or experimental JAX versions where Primitive is hidden
    try:
        from jax._src.core import Primitive
    except ImportError:
        # Fallback for even weirder versions
        Primitive = core.CallPrimitive.__bases__[0]

quantum_circuit_p = Primitive("quantum_circuit")


def execute_circuit(circuit: Circuit, *params: float, backend: str = "statevector") -> jax.Array:
    # If the user passed a single list/array/tuple/sequence as the first arg,
    # and it is indeed a sequence of parameters (not a single scalar array), unpack it.
    if len(params) == 1:
        p0 = params[0]
        if isinstance(p0, (list, tuple)):
            actual_params = p0
        elif hasattr(p0, "ndim") and p0.ndim > 0:
            actual_params = p0
        else:
            actual_params = params
    else:
        actual_params = params

    if backend in ("jax", "singularity", "jax_mps"):
        from superfermion.backends.factory import get_backend
        sim = get_backend(backend)
        p_names = circuit.parameters
        p_dict = {name: val for name, val in zip(p_names, actual_params)}
        bound_circuit = circuit.bind(p_dict)
        return sim.run(bound_circuit, shots=0).statevector

    return quantum_circuit_p.bind(*actual_params, circuit=circuit)


# ───────────────────────────────────────────────────────────
# 2. Abstract Evaluation (Static typing for JAX)
# ───────────────────────────────────────────────────────────

def quantum_circuit_abstract_eval(*params: core.AbstractValue, **kwargs: Any) -> core.ShapedArray:
    """Determines the output shape/type during JAX tracing."""
    circuit: Circuit = kwargs["circuit"]
    n_out = 2**circuit.n_qubits
    return core.ShapedArray((n_out,), jnp.float32)

# Register abstract_eval
if hasattr(quantum_circuit_p, "def_abstract_eval"):
    quantum_circuit_p.def_abstract_eval(quantum_circuit_abstract_eval)
else:
    # Manual registration in older JAX
    from jax.core import abstract_eval_rules
    abstract_eval_rules[quantum_circuit_p] = quantum_circuit_abstract_eval


# ───────────────────────────────────────────────────────────
# 3. Implementation (Concrete execution)
# ───────────────────────────────────────────────────────────

def _quantum_circuit_execution_host(params: Sequence[float], circuit: Circuit) -> np.ndarray:
    """The actual heavy lifting, running on the host (not inside JAX/XLA)."""
    param_names = circuit.parameters
    
    # Bind numeric values (now guaranteed to be concrete floats)
    bind_dict = {name: float(val) for name, val in zip(param_names, params)}
    bound_circuit = circuit.bind(bind_dict)
    
    # Run the circuit
    result = run(bound_circuit, shots=1000)
    
    n = circuit.n_qubits
    out = np.zeros(2**n, dtype=np.float32)
    
    # Try multiple sources for probabilities to avoid zero-gradient bug
    if result.statevector is not None:
        # If we have a statevector, use its probabilities for zero-noise gradient
        sv = np.array(result.statevector)
        out = np.abs(sv)**2
    elif result.probabilities:
        for bitstring, p in result.probabilities.items():
            idx = int(bitstring, 2)
            out[idx] = p
    elif result.counts:
        total = sum(result.counts.values())
        for bitstring, count in result.counts.items():
            idx = int(bitstring, 2)
            out[idx] = count / total
            
    return out.astype(np.float32)

def quantum_circuit_impl(*params: float, **kwargs: Any) -> jax.Array:
    """Actually run the circuit on a backend.
    
    Supports JIT by using pure_callback to run the simulation on the host.
    """
    circuit: Circuit = kwargs["circuit"]
    n_out = 2**circuit.n_qubits
    result_shape = jax.ShapeDtypeStruct((n_out,), jnp.float32)
    
    # We use pure_callback to allow this "opaque" Python code to run 
    # even when under jax.jit. It handles the transition from 
    # abstract tracers to concrete values.
    # We wrap the host call to pass the circuit via a closure or partial.
    host_fn = lambda p: _quantum_circuit_execution_host(p, circuit)
    
    # Use jnp.stack if params is a sequence of tracers to avoid direct jnp.array(tracers)
    params_tensor = jnp.stack(params) if params else jnp.array([])
    return jax.pure_callback(host_fn, result_shape, params_tensor)

# Register implementation
if hasattr(quantum_circuit_p, "def_impl"):
    quantum_circuit_p.def_impl(quantum_circuit_impl)
else:
    from jax.core import impl_rules
    impl_rules[quantum_circuit_p] = quantum_circuit_impl


# ───────────────────────────────────────────────────────────
# 4. JVP (Jacobian-Vector Product) — Autodiff Rule
# ───────────────────────────────────────────────────────────

def quantum_circuit_jvp(primals: Sequence[Any], tangents: Sequence[Any], **kwargs: Any) -> tuple[Any, Any]:
    """Compute ∇f(x)·v using the Parameter-Shift Rule.
    
    This implementation returns traceable JAX expressions.
    """
    # Use the primitive itself for the primal output to remain traceable
    # But usually inside a JVP we just return the primal_out computed by the impl
    # However, to be safe and efficient in JAX:
    primal_out = execute_circuit(kwargs["circuit"], *primals)
    
    tangent_out = jnp.zeros_like(primal_out)
    
    for i, v in enumerate(tangents):
        # ad.Zero is a special type used by JAX for zero tangents
        # We can check for it without triggering boolean conversion errors
        if isinstance(v, ad.Zero):
            continue
            
        shift = np.pi / 2
        plus_params = list(primals)
        plus_params[i] += shift
        minus_params = list(primals)
        minus_params[i] -= shift
        
        # We call execute_circuit (the wrapper) so that these 
        # calls are also traced as JAX primitive binds.
        f_plus = execute_circuit(kwargs["circuit"], *plus_params)
        f_minus = execute_circuit(kwargs["circuit"], *minus_params)
        
        # ∂f/∂θ_i = (f_plus - f_minus) / 2
        grad_i = (f_plus - f_minus) / 2
        tangent_out = tangent_out + grad_i * v
        
    return primal_out, tangent_out

# Register JVP
try:
    ad.primitive_jvps[quantum_circuit_p] = quantum_circuit_jvp
except (TypeError, AttributeError):
    try:
        ad.primitive_jvps.def_primitive_jvp(quantum_circuit_p)(quantum_circuit_jvp)
    except (TypeError, AttributeError):
        pass


# ───────────────────────────────────────────────────────────
# 5. Batching (VMap) Rule
# ───────────────────────────────────────────────────────────

def quantum_circuit_batcher(vector_args: Sequence[Any], batch_axes: Sequence[int | None], **kwargs: Any) -> tuple[Any, int]:
    """Support jax.vmap by executing the circuit in a loop.
    
    Uses jax.lax.map to process batch elements sequentially.
    This is necessary because the underlying simulation/callback
    may not natively support vectorization.
    """
    circuit: Circuit = kwargs["circuit"]
    # Re-stack the batched arguments so we can map over them.
    # We assume all batched args are on the same axis (0).
    stacked_args = jnp.stack(vector_args, axis=1)

    def single_exec(p):
        # p is a 1D array of parameters for a single circuit instance
        return execute_circuit(circuit, *p)

    res = jax.lax.map(single_exec, stacked_args)
    return res, 0
    
# Register batcher
try:
    batching.primitive_batchers[quantum_circuit_p] = quantum_circuit_batcher
except (TypeError, AttributeError):
    try:
        batching.primitive_batchers.def_primitive_batcher(quantum_circuit_p)(quantum_circuit_batcher)
    except (TypeError, AttributeError):
        pass


# ───────────────────────────────────────────────────────────
# 6. Lowering to MLIR (for XLA/JIT)
# ───────────────────────────────────────────────────────────

try:
    mlir.register_lowering(quantum_circuit_p, mlir.lower_fun(quantum_circuit_impl, multiple_results=False))
except (AttributeError, TypeError):
    # Older JAX versions might use a different name or mechanism
    pass


# ───────────────────────────────────────────────────────────
# 7. Helper: sf.grad
# ───────────────────────────────────────────────────────────

def circuit_to_jax(circuit: Circuit, backend: str = "statevector"):
    """Convert a Superfermion Circuit to a JAX-compatible function."""
    def jax_wrapped_circuit(*params):
        return execute_circuit(circuit, *params, backend=backend)
        
    return jax_wrapped_circuit

# ───────────────────────────────────────────────────────────
# 8. Density Matrix JAX Wrapper — custom_jvp for noisy autodiff
# ───────────────────────────────────────────────────────────

def make_dm_expectation(circuit: Circuit, observable, backend_name: str = "density_matrix"):
    """Create a JAX-differentiable function ``f(params) -> expectation`` for DM backend.

    Uses ``jax.custom_vjp`` with ``jax.pure_callback`` so that the quantum
    backend (Rust/NumPy) is called with concrete values even during tracing.

    Args:
        circuit:      Parametric SF circuit.
        observable:   SF observable (SparsePauliOp / Hamiltonian / PauliString).
        backend_name: Backend name (default: 'density_matrix').

    Returns:
        A JAX-traceable function ``f(params) -> scalar`` where ``params`` is a
        1-D JAX array.

    Example::

        f = make_dm_expectation(circuit, obs)
        grad = jax.grad(f)(params)    # works even with noisy DM!
    """
    param_names = list(circuit.parameters)
    n_params = len(param_names)

    def _eval_concrete(params_np):
        """Evaluate expectation with concrete numpy params (runs on host)."""
        from superfermion.backends.factory import get_backend
        backend = get_backend(backend_name)
        p_dict = {n: float(p) for n, p in zip(param_names, params_np)}
        bound = circuit.bind(p_dict)
        return float(backend.expval(bound, observable))

    def _grad_concrete(params_np):
        """Parameter-shift gradient with concrete numpy params."""
        from superfermion.backends.factory import get_backend
        backend = get_backend(backend_name)
        shift = float(np.pi / 2.0)
        grads = np.zeros(n_params, dtype=np.float32)
        for i in range(n_params):
            p_arr = np.array(params_np, dtype=np.float32)

            p_plus = p_arr.copy()
            p_plus[i] += shift
            ev_plus = _eval_concrete(p_plus)

            p_minus = p_arr.copy()
            p_minus[i] -= shift
            ev_minus = _eval_concrete(p_minus)

            grads[i] = 0.5 * (ev_plus - ev_minus)
        return grads

    @jax.custom_vjp
    def _expectation(params):
        # Use pure_callback to call concrete code even during tracing
        result_shape = jax.ShapeDtypeStruct((), jnp.float32)
        return jax.pure_callback(
            lambda p: jnp.array(_eval_concrete(np.asarray(p)), dtype=jnp.float32),
            result_shape,
            params,
        )

    def _fwd(params):
        val = _expectation(params)
        return val, params

    def _bwd(params, g):
        result_shape = jax.ShapeDtypeStruct((n_params,), jnp.float32)
        grads = jax.pure_callback(
            lambda p: jnp.array(_grad_concrete(np.asarray(p)), dtype=jnp.float32),
            result_shape,
            params,
        )
        return (grads * g,)

    _expectation.defvjp(_fwd, _bwd)

    return _expectation


# Convenience wrapper matching old signature
def dm_expectation(circuit: Circuit, observable, params, backend_name: str = "density_matrix"):
    """JAX-differentiable expectation value via density matrix backend.

    Usage::

        # Direct call
        val = dm_expectation(circuit, obs, params)

        # Gradient (use the factory for better JAX compatibility)
        f = make_dm_expectation(circuit, obs)
        grad = jax.grad(f)(params)

        # Or inline
        grad = jax.grad(lambda p: dm_expectation(circuit, obs, p))(params)
    """
    f = make_dm_expectation(circuit, observable, backend_name)
    return f(params)

