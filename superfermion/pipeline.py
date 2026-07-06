"""
Superfermion Pipeline — Declarative workflows for quantum machine learning.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np
import jax.numpy as jnp

from superfermion.data.dataset import Dataset
from superfermion.data.preprocessing import min_max_scale, angle_encoding_transform


class Pipeline:
    """
    A declarative pipeline for chaining data preprocessing, 
    quantum encoding, and model execution.

    Args:
        steps: List of (name, transformation/model) tuples.
    
    Example:
        >>> from superfermion.nn import QuantumLayer
        >>> pipe = Pipeline([
        ...     ('scale', min_max_scale),
        ...     ('quantum', QuantumLayer(n_qubits=4, ansatz=my_ansatz))
        ... ])
        >>> results = pipe.execute(data)
    """

    def __init__(self, steps: Optional[List[Tuple[str, Any]]] = None, name: str = "pipeline"):
        self.steps = steps if steps is not None else []
        self.name = name
        self.metadata: Dict[str, Any] = {}

    def add_stage(self, name: str, step: Any) -> None:
        """Add a processing stage to the pipeline."""
        self.steps.append((name, step))

    def run(self, data: Union[np.ndarray, Dataset], params: Optional[Any] = None) -> Any:
        """Run the pipeline on input data. Alias for execute()."""
        return self.execute(data, params)

    def execute(self, data: Union[np.ndarray, Dataset], params: Optional[Any] = None) -> Any:
        """
        Execute the pipeline on the input data.

        Args:
            data:   Input features (numpy array or Dataset).
            params: Optional model parameters (for flax-based steps).

        Returns:
            The output of the final step in the pipeline.
        """
        current_data = data.X if isinstance(data, Dataset) else data

        for name, step in self.steps:
            if hasattr(step, 'apply') and params is not None:
                # If it's a Flax module or has an apply method
                current_data = step.apply({'params': params}, current_data)
            elif callable(step):
                # If it's a simple transformation function or other callable
                current_data = step(current_data)
            elif hasattr(step, '__call__'):
                # General callable object
                current_data = step(current_data)
            else:
                raise ValueError(f"Step '{name}' is not a valid transformation or model.")

        return current_data

    def save_checkpoint(self, path: str, data: Dict[str, Any]) -> None:
        """Save a checkpoint with pipeline state to JSON."""
        import json
        payload = {
            "name": self.name,
            "metadata": self.metadata,
            "data": data,
        }
        with open(path, 'w') as f:
            json.dump(payload, f)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint from JSON and return the saved data."""
        import json
        with open(path, 'r') as f:
            payload = json.load(f)
        self.name = payload.get("name", self.name)
        self.metadata = payload.get("metadata", {})
        return payload.get("data", {})

    def __repr__(self) -> str:
        step_names = " -> ".join([name for name, _ in self.steps])
        return f"Pipeline({step_names})"


def make_pipeline(*steps: Any) -> Pipeline:
    """Helper to create a pipeline from a list of unnamed steps."""
    named_steps = [(f"step_{i}", step) for i, step in enumerate(steps)]
    return Pipeline(named_steps)
