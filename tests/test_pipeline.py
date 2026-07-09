import pytest
import torch

from dkg_tlan import (
    CrossLayerCorrelation,
    DynamicKnowledgeGraph,
    TemporalLogicalAttentionNetwork,
)
from dkg_tlan.fusion import GMMAnomalyScorer, mean_reciprocal_rank, precision_at_k


def test_end_to_end_forward_pass():
    torch.manual_seed(0)
    num_nodes, seq_len, input_dim, hidden_dim = 13, 32, 16, 8

    dkg = DynamicKnowledgeGraph(feature_dim=input_dim)
    node_features = torch.randn(num_nodes, input_dim)
    dkg_out = dkg(node_features)
    assert dkg_out["adjacency"].shape == (num_nodes, num_nodes)

    tlan = TemporalLogicalAttentionNetwork(input_dim, hidden_dim)
    window = torch.randn(1, seq_len, input_dim)
    tlan_out = tlan(window, node_features, dkg_out["adjacency"])
    assert tlan_out["h_temp"].shape == (1, seq_len, 2 * hidden_dim)
    assert tlan_out["h_log"].shape == (num_nodes, hidden_dim)

    fusion = CrossLayerCorrelation(
        temp_dim=2 * hidden_dim, log_dim=hidden_dim, adj_dim=num_nodes
    )
    h_log_seq = tlan_out["h_log"].mean(0, keepdim=True).expand(seq_len, -1).unsqueeze(0)
    adj_summary = dkg_out["adjacency"].mean(0, keepdim=True).expand(seq_len, -1).unsqueeze(0)
    h_final = fusion(tlan_out["h_temp"], h_log_seq, adj_summary)
    assert h_final.shape == tlan_out["h_temp"].shape

    scorer = GMMAnomalyScorer(feature_dim=h_final.shape[-1])
    scores = scorer.score(h_final)
    assert scores.shape == h_final.shape[:-1]
    scorer.update(h_final)


def test_root_cause_metrics():
    ranks = torch.tensor([1, 2, 1, 5, 3], dtype=torch.float32)
    assert precision_at_k(ranks, k=1) == pytest.approx(0.4)
    assert precision_at_k(ranks, k=3) == pytest.approx(0.8)
    mrr = mean_reciprocal_rank(ranks)
    assert 0.0 < mrr <= 1.0
