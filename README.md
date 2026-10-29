# DKG-TLAN

**Dynamic Knowledge Graph with Temporal Logical Attention Network**: a hybrid
framework for anomaly detection and root-cause localization in cloud-native
microservice environments.

[![Paper: IEEE Xplore](https://img.shields.io/badge/paper-IEEE%20Xplore-00629B)](https://ieeexplore.ieee.org/document/11724336)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICoICI69824.2026.11724336-blue)](https://doi.org/10.1109/ICoICI69824.2026.11724336)
[![CI](https://github.com/sunilgentyala/DKG-TLAN/actions/workflows/ci.yml/badge.svg)](https://github.com/sunilgentyala/DKG-TLAN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Site: https://sunilgentyala.github.io/DKG-TLAN/

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

## Reproducible synthetic benchmark

`benchmarks/synthetic_eval.py` wires the modules above into an end-to-end,
exactly reproducible classifier (13-node topology, 94.2%/5.8% normal/
anomalous class balance, W=64-timestep sliding windows, d=16 feature
dimensionality, three Chaos-Mesh-style fault classes: pod failure, network
delay, memory stress) and evaluates it against three baselines (a 1D-CNN
"ResNet-50" analog, an LSTM, and a GAT-only GNN), plus four component
ablations (without DKG, without multi-scale kernels, without gating, without
GMM). **This is not the private AWS EKS Sock Shop deployment used for the
paper's Table 1 numbers** — it is a separate, synthetic, fully reproducible
benchmark built to supply class-wise / confusion-matrix / ROC-AUC / PR-AUC /
ablation results for peer review from the released reference implementation.
Run it with:

```bash
python benchmarks/synthetic_eval.py
```

Results (`benchmarks/synthetic_eval_results.json`) at 25 training epochs:

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- |
| Proposed DKG-TLAN | 0.988 | 0.925 | 0.997 | 0.983 |
| DKG-TLAN w/o DKG | 0.996 | 0.975 | 0.994 | 0.986 |
| DKG-TLAN w/o Multi-Scale | 0.994 | 0.958 | 0.998 | 0.981 |
| DKG-TLAN w/o Gating | 0.997 | 0.979 | 0.998 | 0.990 |
| DKG-TLAN w/o GMM | 0.989 | 0.929 | 0.992 | 0.965 |
| ResNet-50 (baseline) | 0.975 | 0.732 | 0.970 | 0.750 |
| LSTM (baseline) | 0.967 | 0.619 | 0.942 | 0.673 |
| GNN (baseline) | 0.994 | 0.949 | 0.994 | 0.981 |

DKG-TLAN clearly beats the ResNet-50 and LSTM baselines. The ablation
comparison is reported honestly: on this i.i.d., single-topology synthetic
task, the ablated variants matched or marginally exceeded the full model's
aggregate metrics, with the gap narrowing substantially as training budget
increased (see `benchmarks/synthetic_eval_log.txt` for the 6/8/12-epoch
runs this converged from). This benchmark does not exercise concept drift
or dynamic topology change, which are the operational conditions DKG's
adaptive adjacency and GMM's EMA drift adaptation are designed for.

## Citation

```bibtex
@inproceedings{mudusu2026dkgtlan,
  title     = {Anomaly Detection by using Dynamic Knowledge Graph with Temporal
               Attention Network},
  author    = {Mudusu, Sunil Kumar and Gentyala, Sunil and Pendyala, Manasa},
  booktitle = {2026 4th International Conference on Intelligent Cyber Physical
               Systems and Internet of Things (ICoICI)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICoICI69824.2026.11724336}
}
```

The paper is published on IEEE Xplore:
[10.1109/ICoICI69824.2026.11724336](https://ieeexplore.ieee.org/document/11724336).
To cite the software itself, use [`CITATION.cff`](CITATION.cff).

## Authors

- Sunil Kumar Mudusu, Church Mutual Insurance Company
- Sunil Gentyala, HCLTech (HCL America Inc.)
- Manasa Pendyala, Oracle

## Security

See [SECURITY.md](SECURITY.md) for how to report a vulnerability.

## License

MIT. See [LICENSE](LICENSE).
