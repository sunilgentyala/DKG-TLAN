# DKG-TLAN

**Dynamic Knowledge Graph with Temporal Logical Attention Network**: a hybrid
framework for anomaly detection and root-cause localization in cloud-native
microservice environments.

[![Paper: Under Submission](https://img.shields.io/badge/paper-under%20submission-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📄 Site: https://sunilgentyala.github.io/DKG-TLAN/

## Overview

Modern microservice platforms scale, fail, and migrate across physical nodes
faster than static dependency models can track. DKG-TLAN combines:

- **Dynamic Knowledge Graph (DKG)**: reconstructs physical-to-virtual
  structural relationships from Istio traces, raw logs, and Kubernetes
  orchestration metadata using an adaptive, sparsified adjacency matrix.
- **Temporal Logical Attention Network (TLAN)**: parallel multi-scale
  Conv1D feature extraction (kernel sizes 3, 5, 7) fused with a dual
  BiLSTM (temporal) + Graph Attention Network (structural) branch.
- **Cross-layer correlation and gated fusion**: a learnable gate balances
  temporal and structural evidence to flag anomalies and localize the
  shared physical hardware bottleneck behind them, rather than stopping at
  the symptomatic logical service.

This repository provides a from-scratch, framework-faithful reference
implementation of the model's core modules (Eq. 1-18 in the paper), intended
for readers who want to inspect or extend the architecture. It is not the
private training/evaluation harness used for the paper's reported results.

## Architecture

```
Istio traces + raw logs + K8s metadata
              │
              ▼
   ┌─────────────────────┐
   │  Dynamic Knowledge   │   adaptive adjacency learning + sparsification
   │  Graph (DKG)         │   (dkg.py)
   └──────────┬───────────┘
              │ adjacency matrix
              ▼
   ┌─────────────────────┐
   │  Multi-Scale Feature │   Conv1D kernels {3,5,7} + softmax fusion
   │  Extraction          │   (tlan.py)
   └──────────┬───────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
  ┌────────┐      ┌──────────┐
  │ BiLSTM │      │   GAT    │   temporal vs. logical/structural branches
  │(h_temp)│      │ (h_log)  │   (tlan.py)
  └───┬────┘      └────┬─────┘
      └───────┬─────────┘
              ▼
   ┌─────────────────────┐
   │ Cross-Layer          │   cosine correlation + learnable gate
   │ Correlation & Fusion │   (fusion.py)
   └──────────┬───────────┘
              ▼
   ┌─────────────────────┐
   │ GMM Anomaly Scoring  │   NLL score + EMA drift adaptation
   └──────────┬───────────┘
              ▼
     anomaly score + root-cause ranking (PR@K, MRR)
```

## Installation

```bash
git clone https://github.com/sunilgentyala/DKG-TLAN.git
cd DKG-TLAN
pip install -e ".[dev]"
```

## Usage

```python
import torch
from dkg_tlan import DynamicKnowledgeGraph, TemporalLogicalAttentionNetwork, CrossLayerCorrelation

num_nodes, seq_len, input_dim, hidden_dim = 13, 32, 16, 8
node_features = torch.randn(num_nodes, input_dim)
window = torch.randn(1, seq_len, input_dim)

dkg = DynamicKnowledgeGraph(feature_dim=input_dim)
adjacency = dkg(node_features)["adjacency"]

tlan = TemporalLogicalAttentionNetwork(input_dim, hidden_dim)
out = tlan(window, node_features, adjacency)  # h_temp, h_log
```

Run the test suite:

```bash
pytest tests/ -v
```

## Reference experimental setup (as reported in the paper)

- Testbed: Sock Shop polyglot microservice application (13 services) on AWS
  EKS, with Istio + Grafana Tempo for tracing, Grafana Loki for log
  aggregation, Prometheus + Scaphandre for node-level metrics, and Chaos
  Mesh for fault injection.
- Load generation: [Locust](https://locust.io/).
- Reported results: 99.75% accuracy, 98.53% precision, 99.64% recall, 98.46%
  F1-score on the Sock Shop benchmark, with a 2.9 ms mean inference time.

## Citation

```
S. K. Mudusu, S. Gentyala, and M. Pendyala,
"Anomaly Detection by using Dynamic Knowledge Graph with Temporal Attention Network,"
under submission, 2026.
```

## Authors

- Sunil Kumar Mudusu, Church Mutual Insurance Company
- Sunil Gentyala, HCLTech (HCL America Inc.)
- Manasa Pendyala, Oracle

## License

MIT. See [LICENSE](LICENSE).
