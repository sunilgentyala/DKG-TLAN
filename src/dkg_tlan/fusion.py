"""Cross-layer correlation, gated fusion, and anomaly scoring.

Implements Eq. 12-16: cosine-similarity correlation between temporal and
logical branches, a learnable gate that balances the two representations,
Gaussian-Mixture anomaly scoring, and an exponential-moving-average update
rule for online adaptation to concept drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayerCorrelation(nn.Module):
    """Gated fusion of temporal (h_temp) and logical/structural (h_log, A_t) branches."""

    def __init__(self, temp_dim: int, log_dim: int, adj_dim: int):
        super().__init__()
        self.gate = nn.Linear(temp_dim + log_dim + adj_dim, temp_dim)
        self.structural_proj = nn.Linear(log_dim + adj_dim, temp_dim)

    def correlation(self, h_temp: torch.Tensor, h_log: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity correlation matrix C_ij (Eq. 12)."""
        return F.cosine_similarity(h_temp.unsqueeze(1), h_log.unsqueeze(0), dim=-1)

    def forward(
        self, h_temp: torch.Tensor, h_log: torch.Tensor, adjacency_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_temp: (batch, T, temp_dim) BiLSTM output.
            h_log:  (batch, T, log_dim)  GAT output, aligned to the same T.
            adjacency_summary: (batch, T, adj_dim) pooled adjacency features.

        Returns:
            h_final: (batch, T, temp_dim) fused representation (Eq. 14).
        """
        gate_input = torch.cat([h_temp, h_log, adjacency_summary], dim=-1)
        g_t = torch.sigmoid(self.gate(gate_input))  # Eq. 13
        structural = self.structural_proj(torch.cat([h_log, adjacency_summary], dim=-1))
        h_final = g_t * h_temp + (1 - g_t) * structural  # Eq. 14
        return h_final


class GMMAnomalyScorer(nn.Module):
    """Gaussian-Mixture negative log-likelihood anomaly score (Eq. 15) with EMA
    parameter adaptation for concept drift (Eq. 16)."""

    def __init__(self, feature_dim: int, num_components: int = 4, momentum: float = 0.05):
        super().__init__()
        self.register_buffer("means", torch.randn(num_components, feature_dim) * 0.01)
        self.register_buffer("log_vars", torch.zeros(num_components, feature_dim))
        self.momentum = momentum

    def score(self, h_final: torch.Tensor) -> torch.Tensor:
        # negative log-likelihood under the current mixture, per Eq. 15
        diffs = h_final.unsqueeze(-2) - self.means  # (..., K, D)
        var = self.log_vars.exp()
        log_prob = -0.5 * ((diffs**2) / var + self.log_vars).sum(-1)
        nll = -torch.logsumexp(log_prob, dim=-1)
        return nll

    @torch.no_grad()
    def update(self, h_final: torch.Tensor) -> None:
        """Online EMA update of mixture means (Eq. 16)."""
        pooled = h_final.reshape(-1, h_final.shape[-1]).mean(dim=0)
        self.means.mul_(1 - self.momentum).add_(self.momentum * pooled)


def precision_at_k(ranks: torch.Tensor, k: int) -> float:
    """PR@K (Eq. 17): fraction of trials where the true root cause is in the top-k."""
    return (ranks <= k).float().mean().item()


def mean_reciprocal_rank(ranks: torch.Tensor) -> float:
    """MRR (Eq. 18)."""
    return (1.0 / ranks.float()).mean().item()
