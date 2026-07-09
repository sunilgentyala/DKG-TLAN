"""Dynamic Knowledge Graph (DKG) construction.

Implements the adaptive adjacency-matrix learning strategy described in the
paper: co-occurrence-weighted dependency scoring (Eq. 1), a learnable
similarity transform (Eq. 2), and threshold-based sparsification (Eq. 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def dependency_weight(component_sequence: torch.Tensor, num_components: int) -> torch.Tensor:
    """Compute the co-occurrence dependency weight w_ij (Eq. 1).

    Args:
        component_sequence: 1D LongTensor of component indices active at each
            time step, shape (T,).
        num_components: total number of distinct components (nodes).

    Returns:
        (num_components, num_components) weight matrix.
    """
    transitions = torch.zeros(num_components, num_components)
    occ = torch.zeros(num_components)
    for t in range(component_sequence.shape[0] - 1):
        i, j = int(component_sequence[t]), int(component_sequence[t + 1])
        transitions[i, j] += 1.0
        occ[i] += 1.0
    occ[-1] += 1.0 if component_sequence.shape[0] else 0.0

    denom = torch.sqrt(occ.unsqueeze(1) * occ.unsqueeze(0)).clamp_min(1e-8)
    return transitions / denom


class AdaptiveAdjacencyLearner(nn.Module):
    """Learnable similarity transform + sparsification (Eq. 2-3)."""

    def __init__(self, feature_dim: int, sparsify_threshold: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.threshold = sparsify_threshold

    def forward(self, component_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            component_features: (N, feature_dim) per-node feature matrix
                (e.g. pooled trace / log embeddings for the current window).

        Returns:
            (N, N) sparsified adjacency matrix.
        """
        proj = torch.tanh(self.proj(component_features))
        similarity = torch.exp(-torch.cdist(proj, proj, p=2))
        sparsified = torch.where(
            similarity >= self.threshold, similarity, torch.zeros_like(similarity)
        )
        return sparsified


class DynamicKnowledgeGraph(nn.Module):
    """End-to-end DKG builder: co-occurrence prior + learned, sparsified adjacency."""

    def __init__(self, feature_dim: int, sparsify_threshold: float = 0.15):
        super().__init__()
        self.adjacency_learner = AdaptiveAdjacencyLearner(feature_dim, sparsify_threshold)

    def forward(self, component_features: torch.Tensor) -> dict:
        adjacency = self.adjacency_learner(component_features)
        return {
            "adjacency": adjacency,
            "num_edges": int((adjacency > 0).sum().item()),
        }
