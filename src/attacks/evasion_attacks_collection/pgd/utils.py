import torch
from tqdm import tqdm

# FIXME where do we use all these?

class Projection:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, a):
        """
        """
        # a = a_matrix.flatten()

        projection = self.projection(a)

        # projection_matrix = projection.view(a_matrix.shape)
        return projection

    def projection(self, a):
        """
        Calculating the projection of 'a' onto a set 'S'
        """
        if torch.isnan(a).any():
            # NaN found in vector a. Replace with zeros.
            a = torch.nan_to_num(a, nan=0.0)

        # Projection onto [0, 1]
        s = torch.clamp(a, min=0, max=1)

        # Check the sum, otherwise project onto simplex
        if torch.sum(s) <= self.eps:
            return s

        # Projection onto simplex
        return self.projection_onto_simplex(s)

    def projection_onto_simplex(self, v):
        """
        Projection of a vector 'v' onto a simplex with a restriction on the sum of elements
        """
        if torch.sum(v) <= self.eps:
            return v

        # Sort the elements of the vector 'v' in descending order
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - self.eps

        # Find the index 'rho' where the projection begins
        mask = u > (cssv / torch.arange(1, len(u) + 1, dtype=v.dtype))
        indices = torch.nonzero(mask).squeeze()

        rho = indices[-1]

        theta = cssv[rho] / (rho.item() + 1)

        # Final projection onto simplex
        proj = torch.clamp(v - theta, min=0)

        # This simplex projection algorithm guarantees that torch.sum(v) <= self.eps.
        # Let's write an assertion taking into account the rule for comparing floating-point numbers.
        assert torch.allclose(torch.sum(proj), torch.tensor(self.eps, dtype=torch.float)) is True

        return proj


def random_sampling(edge_mask: torch.Tensor, budget: int, num_trials: int = 20, model=None,
                  x=None, edge_index=None, y=None, attack_loss=None, task_type=True, target_idx=None):
    """
    Performs random sampling over a probabilistic edge mask to generate discrete binary masks,
    while respecting a budget constraint on the number of modified edges.

    Args:
        edge_mask (torch.Tensor): Probabilistic edge mask of shape [E].
        budget (int): Maximum number of edges allowed to be modified.
        num_trials (int): Number of random sampling trials.
        model: GNN model.
        x, edge_index, y: Graph data.
        attack_loss: attack_loss loss function.
        target_idx: Target node index for node classification.

    Returns:
        best_mask (torch.BoolTensor): Binary edge mask [E]. True values
        indicate that the edge should be removed to improve the attack quality.
    """

    best_loss = float('inf')
    best_mask = None

    for _ in tqdm(range(num_trials), desc="Random sampling", leave=True):
        # Discretization: sample binary mask u_i ∈ {0, 1} from s_i ∈ [0, 1]
        sampled_mask = torch.bernoulli(edge_mask).bool()

        # Apply budget constraint
        if sampled_mask.sum().item() > budget:
            # Since the algorithm works on edge deletion, edges with values 1.0 or close to 1.0 remain.
            # This means that the most influential edge (i.e., when deleting which, the -loss of the model will
            # decrease) has small values in the edge_mask mask. Therefore, topk is taken from -edge_mask.
            bottomk = torch.topk(-edge_mask, budget).indices
            sampled_mask = torch.zeros_like(edge_mask, dtype=torch.bool)
            sampled_mask[bottomk] = True

        # Run the model and compute the loss
        with torch.no_grad():
            out = model(x, edge_index[:, sampled_mask])
            if task_type:
                loss = attack_loss(out, y)
            else:
                loss = attack_loss(out[target_idx], y[target_idx])

        # Update the best mask
        if loss < best_loss:
            best_loss = loss
            best_mask = sampled_mask.clone()

    return best_mask