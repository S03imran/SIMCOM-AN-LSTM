import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import random
from collections import defaultdict
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import TGCN
from sklearn.metrics import roc_auc_score, average_precision_score


# ---- TGCN Model Definition ----
class TGCNModel(torch.nn.Module):
    def __init__(self, node_features, out_channels):
        super(TGCNModel, self).__init__()
        self.tgcn = TGCN(node_features, out_channels)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        # Return node embeddings
        h = self.tgcn(x, edge_index, edge_weight)
        return torch.relu(h)


# ---- Negative edge sampler ----
def sample_negative_edges(all_nodes, existing_edges, num_samples):
    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
            neg_edges.add((u, v))
    return list(neg_edges)


# ---- Build temporal dataset from timestamped edges ----
def build_temporal_dataset(path):
    # Expect file as: u v timestamp
    data = np.loadtxt(path, dtype=int)
    edges_by_time = defaultdict(list)
    all_nodes = set()

    for u, v, t in data:
        edges_by_time[t].append((u, v))
        all_nodes.add(u)
        all_nodes.add(v)

    all_nodes = sorted(list(all_nodes))
    node_count = max(all_nodes) + 1  # Ensure 0..max_id indexing

    edge_indices, edge_weights, features = [], [], []

    # Create one snapshot per timestamp
    for t in sorted(edges_by_time.keys()):
        G = nx.Graph()
        G.add_nodes_from(range(node_count))
        G.add_edges_from(edges_by_time[t])

        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        weight = torch.ones(edge_index.shape[1], dtype=torch.float)

        degrees = np.array([G.degree(n) for n in range(node_count)], dtype=float).reshape(-1, 1)
        feat_t = torch.tensor(degrees, dtype=torch.float)

        edge_indices.append(edge_index)
        edge_weights.append(weight)
        features.append(feat_t)

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, features)
    return dataset, all_nodes


# ---- Train and evaluate TGCN ----
def run_tgcn(dataset, all_nodes, train_ratio, runs=5):
    auc_sum, ap_sum = 0, 0
    valid_runs = 0

    for run in range(runs):
        model = TGCNModel(node_features=1, out_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_len = int(len(dataset.features) * train_ratio)
        if train_len >= len(dataset.features):
            print("Train ratio too high; no test snapshots left.")
            break

        # ---- Training ----
        for epoch in range(5):
            model.train()
            for t in range(train_len):
                optimizer.zero_grad()
                x = dataset.features[t]
                ei = dataset.edge_indices[t]
                ew = dataset.edge_weights[t]

                emb = model(x, ei, ew)

                # Match label size to number of nodes in this snapshot
                labels = torch.zeros((x.shape[0], 1))
                for u, v in ei.t().tolist():
                    labels[u] = 1
                    labels[v] = 1

                logits = torch.sigmoid(model.linear(emb))
                loss = F.binary_cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

        # ---- Evaluation on next snapshot ----
        model.eval()
        with torch.no_grad():
            x = dataset.features[train_len]
            ei = dataset.edge_indices[train_len]
            ew = dataset.edge_weights[train_len]
            emb = model(x, ei, ew)

        test_edges = ei.t().tolist()

        # Collect all training edges
        all_train_edges = set()
        for t in range(train_len):
            all_train_edges.update([tuple(edge) for edge in dataset.edge_indices[t].t().tolist()])

        # Sample negatives equal to positives
        neg_edges = sample_negative_edges(all_nodes, all_train_edges, len(test_edges))
        eval_edges = test_edges + neg_edges
        y_true = [1] * len(test_edges) + [0] * len(neg_edges)

        # Compute edge scores from node embeddings
        y_scores = []
        for u, v in eval_edges:
            score = torch.sigmoid(torch.dot(emb[u], emb[v])).item()
            y_scores.append(score)

        # Skip run if only one class present
        if len(set(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        auc_sum += auc
        ap_sum += ap
        valid_runs += 1

    if valid_runs == 0:
        print("No valid runs: all test splits had only one class.")
        return 0.0, 0.0, 0.0

    avg_auc = auc_sum / valid_runs
    avg_ap = ap_sum / valid_runs

    # Return AP in place of accuracy for the 3rd output as you requested
    return avg_auc, avg_ap, avg_ap


# ---- Main ----
if __name__ == "__main__":
    # Update this path to your dataset
    print("test1")
    dataset_path = "datasets_dynamic/math.txt"
    print("test2")
    dataset, all_nodes = build_temporal_dataset(dataset_path)
    print("test3")
    auc, ap1, ap2 = run_tgcn(dataset, all_nodes, train_ratio=0.7, runs=5)
    print("test4")
    print(dataset)
    print(f"AUC: {auc:.4f}")
    print(f"AUPR: {ap1:.4f}")
    print(f"Avg Precision : {ap2:.4f}")
