"""Reproducible synthetic benchmark for DKG-TLAN.

This script generates a synthetic multi-service telemetry benchmark that
mirrors the structural properties described in the paper (13-node topology,
94.2%/5.8% normal/anomalous class balance, three Chaos-Mesh-style fault
types: pod failure, network delay, memory stress) and trains the released
reference implementation (dkg_tlan.DynamicKnowledgeGraph,
TemporalLogicalAttentionNetwork, CrossLayerCorrelation, GMMAnomalyScorer)
plus three baselines (a 1D-CNN "ResNet-50" analog, an LSTM, and a GAT-only
GNN) end-to-end with a linear classification head.

IMPORTANT: this is NOT the private AWS EKS Sock Shop deployment used to
produce Table 1 in the paper. It is a separate, fully synthetic, exactly
reproducible benchmark built to supply the class-wise / confusion-matrix /
ROC-AUC / PR-AUC / component-ablation results requested in peer review,
using the same released model code. Results here should not be conflated
with the operational Table 1 numbers.

Usage:
    python benchmarks/synthetic_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dkg_tlan.dkg import DynamicKnowledgeGraph
from dkg_tlan.tlan import MultiScaleFeatureExtractor, GraphAttentionLayer
from dkg_tlan.fusion import GMMAnomalyScorer

SEED = 1234
NUM_NODES = 13
WINDOW_LEN = 64          # W: sliding-window length (timesteps)
FEATURE_DIM = 16         # d: per-timestep feature dimensionality
HIDDEN_DIM = 32
NUM_CLASSES = 4          # Normal, Pod Failure, Network Delay, Memory Stress
CLASS_NAMES = ["Normal", "Pod Failure", "Network Delay", "Memory Stress"]
N_WINDOWS = 15000
ANOMALY_RATE = 0.058     # 5.8%, matching the paper's stated corpus balance
TRAIN_FRAC = 0.70        # matches the paper's 70/30 chronological split
EPOCHS = 25
LR = 1e-3

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------
def generate_dataset(n_windows: int):
    labels = np.zeros(n_windows, dtype=np.int64)
    n_anom = int(round(n_windows * ANOMALY_RATE))
    anom_idx = rng.choice(n_windows, size=n_anom, replace=False)
    per_class = np.array_split(rng.permutation(anom_idx), 3)
    for cls, idxs in zip([1, 2, 3], per_class):
        labels[idxs] = cls

    X = np.zeros((n_windows, NUM_NODES, WINDOW_LEN, FEATURE_DIM), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, WINDOW_LEN)
    for i in range(n_windows):
        base_phase = rng.uniform(0, 2 * np.pi, size=(NUM_NODES, 1))
        base = 0.5 * np.sin(t[None, :, None] + base_phase[:, None, :])
        base = np.broadcast_to(base, (NUM_NODES, WINDOW_LEN, FEATURE_DIM)).copy()
        noise = rng.normal(0, 0.15, size=(NUM_NODES, WINDOW_LEN, FEATURE_DIM))
        window = base + noise

        cls = labels[i]
        if cls == 1:  # pod failure: subtle correlated dip in a node and its structural neighbor
            node = rng.integers(0, NUM_NODES)
            onset = rng.integers(WINDOW_LEN // 3, WINDOW_LEN - 5)
            window[node, onset:, :] *= 0.7
            neighbor = (node + 1) % NUM_NODES
            window[neighbor, onset:, :] += rng.normal(0.22, 0.05, size=(WINDOW_LEN - onset, FEATURE_DIM))
        elif cls == 2:  # network delay: small correlated lag across a few nodes
            affected = rng.choice(NUM_NODES, size=3, replace=False)
            shift = rng.integers(5, 15)
            for node in affected:
                window[node] = np.roll(window[node], shift, axis=0)
                window[node, :shift, :] += rng.normal(0.18, 0.04, size=(shift, FEATURE_DIM))
        elif cls == 3:  # memory stress: small, slow drift spread across several nodes
            affected = rng.choice(NUM_NODES, size=4, replace=False)
            drift = np.linspace(0, rng.uniform(0.22, 0.35), WINDOW_LEN)
            chans = rng.choice(FEATURE_DIM, size=FEATURE_DIM // 2, replace=False)
            for node in affected:
                window[node][:, chans] += drift[:, None]
        X[i] = window
    return X, labels


print("Generating synthetic dataset...")
X, y = generate_dataset(N_WINDOWS)
n_train = int(N_WINDOWS * TRAIN_FRAC)
X_train_full, y_train_full = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]  # original, imbalanced test distribution

print(f"Train windows: {n_train}  Test windows (original imbalanced dist.): {len(y_test)}")
print("Train class counts:", {c: int((y_train_full == c).sum()) for c in range(NUM_CLASSES)})
print("Test class counts (imbalanced):", {c: int((y_test == c).sum()) for c in range(NUM_CLASSES)})


def downsample_3to1(X, y):
    """3:1 normal:anomalous downsampling, matching Section 4.1 of the paper."""
    anom_mask = y != 0
    n_anom = anom_mask.sum()
    normal_idx = np.where(~anom_mask)[0]
    keep_normal = rng.choice(normal_idx, size=min(len(normal_idx), n_anom * 3), replace=False)
    keep = np.concatenate([keep_normal, np.where(anom_mask)[0]])
    rng.shuffle(keep)
    return X[keep], y[keep]


X_train, y_train = downsample_3to1(X_train_full, y_train_full)
print(f"Downsampled (3:1) train windows: {len(y_train)}")


# ---------------------------------------------------------------------
# Shared temporal-branch input: mean over nodes -> (batch, T, d) log-stream proxy
# ---------------------------------------------------------------------
NODE_FEAT_DIM = 3 * FEATURE_DIM  # first-half mean, second-half mean, std -- preserves onset/co-movement signal


def to_tensors(X):
    window_stream = X.mean(axis=1)          # (n, T, d)  -- pooled multi-node log stream
    half = WINDOW_LEN // 2
    first_half_mean = X[:, :, :half, :].mean(axis=2)   # (n, N, d)
    second_half_mean = X[:, :, half:, :].mean(axis=2)  # (n, N, d)
    std = X.std(axis=2)                                # (n, N, d)
    node_features = np.concatenate([first_half_mean, second_half_mean, std], axis=-1)  # (n, N, 3d)
    return torch.tensor(window_stream), torch.tensor(node_features)


Xtr_stream, Xtr_nodes = to_tensors(X_train)
Xte_stream, Xte_nodes = to_tensors(X_test)
ytr = torch.tensor(y_train)
yte = torch.tensor(y_test)


# ---------------------------------------------------------------------
# Proposed DKG-TLAN classifier, with ablation switches
# ---------------------------------------------------------------------
class DKGTLANClassifier(nn.Module):
    def __init__(self, use_dkg=True, use_multiscale=True, use_gate=True, use_gmm=True):
        super().__init__()
        self.use_dkg = use_dkg
        self.use_gate = use_gate
        self.use_gmm = use_gmm

        self.dkg = DynamicKnowledgeGraph(feature_dim=NODE_FEAT_DIM)
        kernel_sizes = (3, 5, 7) if use_multiscale else (5,)
        self.multi_scale = MultiScaleFeatureExtractor(FEATURE_DIM, HIDDEN_DIM, kernel_sizes=kernel_sizes)
        self.bilstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.gat = GraphAttentionLayer(NODE_FEAT_DIM, HIDDEN_DIM)

        temp_dim = 2 * HIDDEN_DIM
        log_dim = HIDDEN_DIM
        if use_gate:
            self.gate = nn.Linear(temp_dim + log_dim + 1, temp_dim)
            self.structural_proj = nn.Linear(log_dim + 1, temp_dim)
        else:
            self.concat_proj = nn.Linear(temp_dim + log_dim, temp_dim)

        self.gmm = GMMAnomalyScorer(feature_dim=temp_dim) if use_gmm else None
        self.classifier = nn.Linear(temp_dim, NUM_CLASSES)

    def forward(self, window, node_features):
        # window: (1, T, d), node_features: (1, N, d)
        nf = node_features.squeeze(0)  # (N, d)
        if self.use_dkg:
            adjacency = self.dkg(nf)["adjacency"]
        else:
            adjacency = torch.ones(NUM_NODES, NUM_NODES) / NUM_NODES

        fused = self.multi_scale(window)             # (1, T, H)
        h_temp, _ = self.bilstm(fused)                # (1, T, 2H)
        h_log_nodes = self.gat(nf, adjacency)          # (N, H)
        h_log_pooled = h_log_nodes.mean(dim=0, keepdim=True)   # (1, H)
        T = h_temp.shape[1]
        h_log = h_log_pooled.unsqueeze(1).expand(1, T, -1)     # (1, T, H)
        adj_summary = adjacency.mean().view(1, 1, 1).expand(1, T, 1)  # (1, T, 1)

        if self.use_gate:
            gate_in = torch.cat([h_temp, h_log, adj_summary], dim=-1)
            g = torch.sigmoid(self.gate(gate_in))
            structural = self.structural_proj(torch.cat([h_log, adj_summary], dim=-1))
            h_final = g * h_temp + (1 - g) * structural
        else:
            h_final = self.concat_proj(torch.cat([h_temp, h_log], dim=-1))

        pooled = h_final.mean(dim=1)   # (1, 2H)
        logits = self.classifier(pooled)

        aux_loss = None
        if self.use_gmm:
            nll = self.gmm.score(pooled)
            aux_loss = nll.mean()
        return logits, aux_loss


class ResNet1DBaseline(nn.Module):
    """1D-CNN with residual blocks -- 'ResNet-50' analog for time-series input."""
    def __init__(self):
        super().__init__()
        def block(c_in, c_out):
            # GroupNorm (not BatchNorm1d) -- training uses batch_size=1 forward passes,
            # under which BatchNorm1d's per-batch statistics are degenerate/unstable.
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.ReLU(),
                nn.Conv1d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
            )
        self.stem = nn.Conv1d(FEATURE_DIM, HIDDEN_DIM, 3, padding=1)
        self.b1 = block(HIDDEN_DIM, HIDDEN_DIM)
        self.b2 = block(HIDDEN_DIM, HIDDEN_DIM)
        self.b3 = block(HIDDEN_DIM, HIDDEN_DIM)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, window, node_features):
        x = window.transpose(1, 2)  # (1, d, T)
        x = F.relu(self.stem(x))
        x = F.relu(x + self.b1(x))
        x = F.relu(x + self.b2(x))
        x = F.relu(x + self.b3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x), None


class LSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, window, node_features):
        out, (h_n, _) = self.lstm(window)
        return self.fc(h_n[-1]), None


class GNNBaseline(nn.Module):
    """GAT-only, structure without any temporal modelling."""
    def __init__(self):
        super().__init__()
        self.dkg = DynamicKnowledgeGraph(feature_dim=NODE_FEAT_DIM)
        self.gat = GraphAttentionLayer(NODE_FEAT_DIM, HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, window, node_features):
        nf = node_features.squeeze(0)
        adjacency = self.dkg(nf)["adjacency"]
        h = self.gat(nf, adjacency).mean(dim=0, keepdim=True)
        return self.fc(h), None


# ---------------------------------------------------------------------
# Train / eval loop (per-sample forward passes; models are tiny)
# ---------------------------------------------------------------------
def train_model(model, epochs=EPOCHS, gmm_weight=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = Xtr_stream.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        opt.zero_grad()
        for step, i in enumerate(perm):
            window = Xtr_stream[i:i + 1]
            nodes = Xtr_nodes[i:i + 1]
            label = ytr[i:i + 1]
            logits, aux = model(window, nodes)
            loss = F.cross_entropy(logits, label)
            if aux is not None:
                loss = loss + gmm_weight * aux
            loss = loss / 16
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % 16 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()
        print(f"  epoch {ep + 1}/{epochs}  avg loss {total_loss / n:.4f}")
    return model


@torch.no_grad()
def evaluate(model, X_stream, X_nodes, y_true):
    model.eval()
    n = X_stream.shape[0]
    all_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for i in range(n):
        logits, _ = model(X_stream[i:i + 1], X_nodes[i:i + 1])
        all_probs[i] = F.softmax(logits, dim=-1).numpy()
    y_pred = all_probs.argmax(axis=1)
    y_true_np = y_true.numpy()

    acc = accuracy_score(y_true_np, y_pred)
    prec = precision_score(y_true_np, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true_np, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true_np, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred, labels=list(range(NUM_CLASSES)))

    y_true_bin = np.eye(NUM_CLASSES)[y_true_np]
    try:
        roc_auc = roc_auc_score(y_true_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")
    pr_auc = average_precision_score(y_true_bin, all_probs, average="macro")

    per_class = {}
    for c in range(NUM_CLASSES):
        yb = (y_true_np == c).astype(int)
        pb = (y_pred == c).astype(int)
        per_class[CLASS_NAMES[c]] = {
            "precision": float(precision_score(yb, pb, zero_division=0)),
            "recall": float(recall_score(yb, pb, zero_division=0)),
            "f1": float(f1_score(yb, pb, zero_division=0)),
            "support": int(yb.sum()),
        }

    return {
        "accuracy": float(acc), "precision_macro": float(prec), "recall_macro": float(rec),
        "f1_macro": float(f1), "roc_auc_macro": float(roc_auc), "pr_auc_macro": float(pr_auc),
        "confusion_matrix": cm.tolist(), "per_class": per_class,
    }


results = {}

configs = {
    "Proposed_DKG-TLAN": dict(use_dkg=True, use_multiscale=True, use_gate=True, use_gmm=True),
    "DKG-TLAN_wo_DKG": dict(use_dkg=False, use_multiscale=True, use_gate=True, use_gmm=True),
    "DKG-TLAN_wo_MultiScale": dict(use_dkg=True, use_multiscale=False, use_gate=True, use_gmm=True),
    "DKG-TLAN_wo_Gating": dict(use_dkg=True, use_multiscale=True, use_gate=False, use_gmm=True),
    "DKG-TLAN_wo_GMM": dict(use_dkg=True, use_multiscale=True, use_gate=True, use_gmm=False),
}

for name, cfg in configs.items():
    print(f"\n=== Training {name} ===")
    torch.manual_seed(SEED)
    model = DKGTLANClassifier(**cfg)
    train_model(model)
    res_downsampled = evaluate(model, Xte_stream, Xte_nodes, yte)
    results[name] = res_downsampled
    print(f"  {name}: acc={res_downsampled['accuracy']:.4f} f1={res_downsampled['f1_macro']:.4f} "
          f"roc_auc={res_downsampled['roc_auc_macro']:.4f} pr_auc={res_downsampled['pr_auc_macro']:.4f}")

for name, cls in [("ResNet-50", ResNet1DBaseline), ("LSTM", LSTMBaseline), ("GNN", GNNBaseline)]:
    print(f"\n=== Training baseline {name} ===")
    torch.manual_seed(SEED)
    model = cls()
    train_model(model)
    res = evaluate(model, Xte_stream, Xte_nodes, yte)
    results[name] = res
    print(f"  {name}: acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
          f"roc_auc={res['roc_auc_macro']:.4f} pr_auc={res['pr_auc_macro']:.4f}")

out_path = Path(__file__).resolve().parent / "synthetic_eval_results.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {
            "num_nodes": NUM_NODES, "window_len": WINDOW_LEN, "feature_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM, "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
            "n_windows": N_WINDOWS, "anomaly_rate": ANOMALY_RATE, "train_frac": TRAIN_FRAC,
            "epochs": EPOCHS, "seed": SEED,
        },
        "results_on_original_imbalanced_test_set": results,
    }, f, indent=2)

print(f"\nSaved results to {out_path}")
