"""Temporal Logical Attention Network (TLAN).

Multi-scale feature extraction (Eq. 4-6) followed by joint temporal (BiLSTM,
Eq. 7) and logical/structural (GAT, Eq. 8-10) modelling, fused into a single
representation (Eq. 11).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """Three parallel Conv1D kernels (K1=3, K2=5, K3=7) with softmax-attention fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window: (batch, seq_len, input_dim) sliding-window log embeddings.

        Returns:
            (batch, seq_len, hidden_dim) fused multi-scale representation.
        """
        x = window.transpose(1, 2)  # (batch, input_dim, seq_len)
        scale_feats = [conv(x).transpose(1, 2) for conv in self.convs]  # each (b, T, h)
        stacked = torch.stack(scale_feats, dim=0)  # (S, b, T, h)

        scores = self.attn_v(torch.tanh(self.attn_w(stacked))).squeeze(-1)  # (S, b, T)
        alpha = F.softmax(scores, dim=0).unsqueeze(-1)  # (S, b, T, 1)
        fused = (alpha * stacked).sum(dim=0)  # (b, T, h)
        return fused


class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer (Eq. 8-10)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, in_dim)
            adjacency: (N, N) binary/weighted adjacency (0 = no edge).

        Returns:
            (N, out_dim) updated node representations.
        """
        h = self.W_h(node_features)  # (N, out_dim)
        n = h.shape[0]
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        e = F.leaky_relu(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (N, N)

        mask = adjacency > 0
        e = e.masked_fill(~mask, float("-inf"))
        beta = F.softmax(e, dim=1)
        beta = torch.nan_to_num(beta, nan=0.0)

        h_log = torch.sigmoid(beta @ h)
        return h_log


class TemporalLogicalAttentionNetwork(nn.Module):
    """BiLSTM (temporal) + GAT (logical/structural) dual-branch encoder."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.multi_scale = MultiScaleFeatureExtractor(input_dim, hidden_dim)
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.gat = GraphAttentionLayer(input_dim, hidden_dim)

    def forward(
        self, window: torch.Tensor, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict:
        fused = self.multi_scale(window)  # (batch, T, hidden)
        h_temp, _ = self.bilstm(fused)  # (batch, T, 2*hidden)  -- Eq. 7
        h_log = self.gat(node_features, adjacency)  # (N, hidden)          -- Eq. 8-10
        return {"h_temp": h_temp, "h_log": h_log}
