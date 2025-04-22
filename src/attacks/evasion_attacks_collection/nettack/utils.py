import torch
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, k_hop_subgraph
from torch_sparse import SparseTensor, matmul
import math
import random
from tqdm import tqdm


class NettackSurrogate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, edge_index, x):
        adj_squared = self.preprocess_adjacency(edge_index, x.size(0))
        return matmul(adj_squared, x @ self.W)

    @staticmethod
    def preprocess_adjacency(edge_index, num_nodes):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = SparseTensor(row=row, col=col, value=norm, sparse_sizes=(num_nodes, num_nodes))
        return matmul(adj, adj)

    def train_model(self, x, edge_index, y, train_ratio=0.1, epochs=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        num_nodes = x.size(0)
        all_indices = list(range(num_nodes))
        random.shuffle(all_indices)
        train_size = int(train_ratio * num_nodes)
        train_indices = torch.tensor(all_indices[:train_size], dtype=torch.long, device=x.device)
        for _ in tqdm(range(epochs), desc="surrogate model train ..."):
            optimizer.zero_grad()
            out = self.forward(edge_index, x)
            loss = torch.nn.functional.cross_entropy(out[train_indices], y[train_indices])
            loss.backward()
            optimizer.step()

    def get_logits_for_node(self, x, edge_index, node_idx):
        self.eval()
        with torch.no_grad():
            logits = self.forward(edge_index, x)
        return logits[node_idx]


def powerlaw_log_likelihood(alpha, degrees, d_min=2):
    degrees = degrees[degrees >= d_min].float()
    if len(degrees) == 0:
        return -float('inf')
    return len(degrees) * math.log(alpha) + alpha * len(degrees) * math.log(d_min) - (alpha + 1) * degrees.log().sum().item()


def estimate_powerlaw_alpha(degrees, d_min=2):
    degrees = degrees[degrees >= d_min].float()
    if len(degrees) == 0:
        return 2.0  # arbitrary fallback
    return 1 + len(degrees) / (degrees.log().sum().item() - len(degrees) * math.log(d_min - 0.5))


class NettackAttack:
    def __init__(self, model, x, edge_index, num_classes, target_node, direct=True, depth=None, delta_cutoff=0.004):
        self.model = model
        self.x = x.clone().detach()
        self.edge_index = edge_index.clone().detach()
        self.num_classes = num_classes
        self.target_node = target_node
        self.direct = direct
        self.depth = depth if not direct else 1
        self.device = x.device
        self.original_logits = model.get_logits_for_node(x, edge_index, target_node)
        self.true_label = self.original_logits.argmax().item()

        self.degree = torch.bincount(edge_index[0], minlength=x.size(0))
        self.delta_cutoff = delta_cutoff

        if not direct:
            self.influencer_nodes = self._get_influencers()

        self.feature_cooccurrence = (x.T @ x > 0).float()
        self.d_min = 2
        self.alpha_orig = estimate_powerlaw_alpha(self.degree, self.d_min)
        self.ll_orig = powerlaw_log_likelihood(self.alpha_orig, self.degree, self.d_min)

    def _get_influencers(self):
        node_idx, edge_idx, _, _ = k_hop_subgraph(
            self.target_node, self.depth, self.edge_index, relabel_nodes=False
        )
        influencers = node_idx[node_idx != self.target_node]
        return influencers.tolist()

    def strongest_wrong_class(self):
        logits = self.model.get_logits_for_node(self.x, self.edge_index, self.target_node)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[self.true_label] = False
        return logits[mask].argmax().item()

    def compute_feature_grad(self, node):
        self.model.eval()
        x = self.x.clone().detach().requires_grad_(True)
        logits = self.model.forward(self.edge_index, x)[self.target_node]
        loss = logits[self.true_label] - logits[self.strongest_wrong_class()]
        loss.backward()
        return x.grad[node]

    def perturb_feature(self):
        candidates = [self.target_node] if self.direct else self.influencer_nodes
        best_score = float('inf')
        best_node = None
        best_feat = None
        for node in candidates:
            grad = self.compute_feature_grad(node)
            node_features = self.x[node]
            cooc_mask = self.feature_cooccurrence[node_features.bool()].sum(dim=0) > 0
            flip_mask = (((node_features == 0) & (grad < 0)) | ((node_features == 1) & (grad > 0))) & cooc_mask
            scores = torch.abs(grad[flip_mask])
            indices = torch.where(flip_mask)[0]
            if len(scores) == 0:
                continue
            top_score = scores.max().item()
            if top_score < best_score:
                best_score = top_score
                best_feat = indices[scores.argmax()]
                best_node = node
        return best_node, best_feat, best_score

    def perturb_edge(self):
        if not self.direct:
            sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
                self.target_node, self.depth, self.edge_index, relabel_nodes=False)
        else:
            sub_edge_index = self.edge_index
        num_nodes = self.x.size(0)
        best_score = float('inf')
        best_from = None
        for i in range(sub_edge_index.size(1)):
            u, v = sub_edge_index[:, i]
            if self.degree[u] <= 1 or self.degree[v] <= 1:
                continue  # avoid singleton nodes
            mask = ~((self.edge_index[0] == u) & (self.edge_index[1] == v))
            new_edge_index = self.edge_index[:, mask]
            new_degree = self.degree.clone()
            new_degree[u] -= 1
            new_degree[v] -= 1
            alpha_new = estimate_powerlaw_alpha(new_degree, self.d_min)
            ll_new = powerlaw_log_likelihood(alpha_new, new_degree, self.d_min)
            ll_ratio = -2 * self.ll_orig + 2 * ll_new
            if ll_ratio > self.delta_cutoff:
                continue
            logits = self.model.get_logits_for_node(self.x, new_edge_index, self.target_node)
            score = logits[self.true_label] - logits[self.strongest_wrong_class()]
            if score < best_score:
                best_score = score
                best_from = (u.item(), v.item())
        return best_from, best_score

    def attack(self, budget, mode="both"):
        print(f"Starting Nettack attack on node {self.target_node} (label={self.true_label})")
        applied = 0
        attempt = 0
        print(f"Starting Nettack attack on node {self.target_node} (label={self.true_label})")
        while applied < budget:
            print(f"- -- Perturbation{applied + 1} / {budget}(Attempt{attempt + 1}) ---")
            best_node, best_feature, feature_score = (None, None, float('inf'))
            best_edge, edge_score = (None, float('inf'))

            if mode in ["both", "feature"]:
                best_node, best_feature, feature_score = self.perturb_feature()

            if mode in ["both", "structure"]:
                best_edge, edge_score = self.perturb_edge()

            if best_node is not None and (feature_score <= edge_score or mode == "feature"):
                print(f"Feature perturbation: flipped feature {best_feature.item()} on node {best_node}")
                self.x[best_node, best_feature] = 1 - self.x[best_node, best_feature]
                applied += 1
            elif best_edge is not None:
                # Apply edge removal
                print(f"Edge perturbation: removed edge {best_edge}")
                u, v = best_edge
                mask = ~((self.edge_index[0] == u) & (self.edge_index[1] == v))
                self.edge_index = self.edge_index[:, mask]
                self.degree[u] -= 1
                self.degree[v] -= 1
                applied += 1
