"""
Optimization Gallery — Combinatorial problems for QAOA testing.
"""

from typing import List, Tuple
import random
import numpy as np

class OptimizationGallery:
    """Standard optimization instances."""
    
    @staticmethod
    def max_cut_graph(n_nodes: int, edge_prob: float = 0.5) -> List[Tuple[int, int, float]]:
        """Generates an Erdos-Renyi graph for Max-Cut."""
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < edge_prob:
                    edges.append((i, j, 1.0))
        return edges

    @staticmethod
    def portfolio_instance(n_assets: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates mock returns and covariance for Portfolio Optimization."""
        mu = np.random.uniform(0.01, 0.1, n_assets)
        # Positive semi-definite matrix
        A = np.random.rand(n_assets, n_assets)
        sigma = A.T @ A
        return mu, sigma
