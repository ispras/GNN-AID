import torch
import random
from torch.nn import functional as F
from attacks.poison_attacks import PoisonAttacker

from torch_geometric.utils import to_dense_adj


class CLGAAttack(PoisonAttacker):
    name = "CLGAAttack"

    def __init__(self, num_nodes, feature_shape, encoder, augmentation_set, threshold, device="cpu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_shape = feature_shape
        self.encoder = encoder  # Differentiable encoder (e.g., GCN)
        self.augmentation_set = augmentation_set  # Set of augmentation methods
        self.threshold = threshold  # Maximum number of edge changes
        self.device = device

        self.modified_adj = None
        self.augmented_graph = None

    def attack(self, adj_matrix, features):
        """
        Execute the CLGA attack on the graph to maximize contrastive loss.
        """
        adj_matrix = to_dense_adj(adj_matrix).squeeze()
        current_adj = adj_matrix.clone().to(self.device)
        for iteration in range(self.threshold):
            gradients_sum = torch.zeros_like(current_adj)

            for _ in range(len(self.augmentation_set)):
                # Generate augmented views
                t1, t2 = random.sample(self.augmentation_set, 2)
                adj_view1, features_view1 = t1(current_adj, features)
                adj_view2, features_view2 = t2(current_adj, features)

                # Forward pass and compute contrastive loss
                embeddings1 = self.encoder(adj_view1, features_view1)
                embeddings2 = self.encoder(adj_view2, features_view2)
                loss = self.contrastive_loss(embeddings1, embeddings2)

                # Backpropagate to compute gradients
                adj_grad1 = torch.autograd.grad(loss, adj_view1, retain_graph=True)[0]
                adj_grad2 = torch.autograd.grad(loss, adj_view2, retain_graph=True)[0]
                gradients_sum += adj_grad1 + adj_grad2

            # Flip the edge with the largest gradient
            max_gradient_index = torch.argmax(gradients_sum.abs())
            row, col = divmod(max_gradient_index, current_adj.shape[1])
            current_adj[row, col] = 1 - current_adj[row, col]  # Flip edge
            current_adj[col, row] = current_adj[row, col]  # Ensure symmetry

            # Save updated adjacency
            self.modified_adj = current_adj.detach()

    def contrastive_loss(self, embeddings1, embeddings2):
        """
        Compute the contrastive loss based on two embeddings.
        """
        pos_loss = F.cosine_similarity(embeddings1, embeddings2).mean()
        neg_loss = F.cosine_similarity(embeddings1, embeddings2.roll(shifts=1, dims=0)).mean()
        return -pos_loss + neg_loss
