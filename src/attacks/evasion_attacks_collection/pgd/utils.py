import torch
from tqdm import tqdm


class Projection:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, a_matrix):
        """
        a_matrix: torch.tensor((N,N))
        """
        a = a_matrix.flatten()

        projection = self.projection(a)

        projection_matrix = projection.view(a_matrix.shape)
        return projection_matrix

    def projection(self, a):
        """
        Calculating the projection of 'a' onto a set 'S'
        """
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


# TODO check name of variables in RandomSampling Algorithm
class RandomSampling:
    def __init__(self, K, eps, A, attack_loss, model, data):
        """Random sampling from probabilistic to binary topology perturbation"""
        self.K = K
        # TODO add condition (1^T, s) <= eps on result
        self.eps = eps
        self.A = A
        self.attack_loss = attack_loss
        self.model = model
        self.data = data

    def __call__(self, mask):
        u_list = []
        for k in tqdm(range(self.K), desc="Random sampling", leave=True):
            random_matrix = torch.rand_like(mask)
            comparison_matrix = random_matrix < mask
            u = comparison_matrix.to(dtype=torch.float32)
            u_list.append(u)

        print("Please wait...")
        best_u = None
        best_f_value = float('inf')

        for u in u_list:
            preds = self.model(self.data.x, self.A - self.A * mask)
            f_value = self.attack_loss(preds, self.data.y)
            if f_value < best_f_value:
                best_f_value = f_value
                best_u = u
        return best_u