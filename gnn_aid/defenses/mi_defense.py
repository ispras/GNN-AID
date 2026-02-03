import torch
import torch.nn.functional as F

from typing import Literal, Type, Any
from gnn_aid.defenses.defense_base import Defender


class MIDefender(
    Defender
):
    """ Base class for all membership inference (MI) defense methods.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

    def pre_batch(
            self,
            **kwargs
    ):
        pass

    def post_batch(
            self,
            **kwargs
    ):
        pass


class EmptyMIDefender(MIDefender):
    """ Just a stub for MI defense.
    """
    name = "EmptyMIDefender"

    def pre_batch(
            self,
            **kwargs
    ):
        pass

    def post_batch(
            self,
            **kwargs
    ):
        pass


class NoiseMIDefender(MIDefender):
    name = "NoiseMIDefender"

    def __init__(
            self,
            noise_type: Literal["reverse_sigmoid", "random", "none"] = "reverse_sigmoid",
            beta: float = 0.5,
            gamma: float = 1.0,
            noise_scale: float = 0.1,
            temperature: float = 2.0,
            **kwargs
    ):
        """
        Membership Inference defense through logit perturbation.

        Args:
            noise_type: Type of noise to apply ('reverse_sigmoid', 'random', or 'none')
            beta: Magnitude parameter for reverse sigmoid
            gamma: Convergence parameter for reverse sigmoid
            noise_scale: Scale of random noise when noise_type='random'
            temperature: Temperature for softmax when converting probs back to logits
        """
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.beta = beta
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.temperature = temperature

        if noise_type not in ["reverse_sigmoid", "random", "none"]:
            raise ValueError(f"Invalid noise_type: {noise_type}")

    def _apply_reverse_sigmoid(
            self,
            logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply reverse sigmoid perturbation in logit space."""
        # Convert to probabilities temporarily for perturbation
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Apply reverse sigmoid perturbation
        perturbed_logits = self.gamma * logits
        perturbed_probs = torch.sigmoid(perturbed_logits)
        r = self.beta * (perturbed_probs - 0.5)

        # Apply perturbation and normalize probabilities
        perturbed_probs = probs - r
        perturbed_probs = torch.clamp(perturbed_probs, min=1e-10, max=1.0)
        perturbed_probs = perturbed_probs / perturbed_probs.sum(dim=-1, keepdim=True)

        # Convert back to logits using inverse softmax with temperature
        return torch.log(perturbed_probs) * self.temperature

    def _apply_uniform_noise(
            self,
            logits: torch.Tensor
    ) -> torch.Tensor:
        """Add random noise directly to logits while preserving order."""
        # Get class rankings to preserve order
        ranks = logits.argsort(dim=-1, descending=True)

        # Add noise and sort back to original order
        noise = torch.randn_like(logits) * self.noise_scale
        perturbed_logits = logits + noise

        # Restore original ranking to preserve accuracy
        for i in range(len(ranks)):
            perturbed_logits[i] = perturbed_logits[i][ranks[i]].sort(descending=True).values

        return perturbed_logits

    def post_batch(
            self,
            model_manager: Type[Any],
            batch: torch.tensor,
            **kwargs
    ) -> dict:
        """
        Apply noise to model logits and compute loss on modified logits.

        Returns:
            Dictionary containing:
            - 'outputs': Modified logits
            - 'loss': Loss computed on modified logits
        """
        logits = model_manager.gnn(batch.x, batch.edge_index)

        if self.noise_type == "reverse_sigmoid":
            modified_logits = self._apply_reverse_sigmoid(logits)
        elif self.noise_type == "random":
            modified_logits = self._apply_random_noise(logits)
        else:  # "none"
            modified_logits = logits

        modified_loss = model_manager.loss_function(modified_logits, batch.y)

        return {
            "outputs": modified_logits,
            "loss": modified_loss
        }


class NoiseMILinkDefender(MIDefender):
    """
    MI defense for Link Prediction tasks via edge logit perturbation
    """
    name = "NoiseMILinkDefender"

    def __init__(
            self,
            noise_type: Literal["reverse_sigmoid", "random", "none"] = "reverse_sigmoid",
            beta: float = 0.3,
            gamma: float = 0.8,
            noise_scale: float = 0.2,
            temperature: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.beta = beta
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.temperature = temperature

        if noise_type not in ["reverse_sigmoid", "random", "none"]:
            raise ValueError(f"Invalid noise_type: {noise_type}")

    def _apply_reverse_sigmoid_binary(
            self,
            edge_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse sigmoid perturbation for binary classification (link prediction)
        """
        probs = torch.sigmoid(edge_logits / self.temperature)

        perturbed_temp_logits = self.gamma * edge_logits
        perturbed_temp_probs = torch.sigmoid(perturbed_temp_logits)
        r = self.beta * (perturbed_temp_probs - 0.5)

        perturbed_probs = probs - r
        perturbed_probs = torch.clamp(perturbed_probs, min=1e-7, max=1.0 - 1e-7)

        perturbed_logits = torch.logit(perturbed_probs, eps=1e-7) * self.temperature
        return perturbed_logits

    def _apply_random_noise(
            self,
            edge_logits: torch.Tensor
    ) -> torch.Tensor:
        """Add Gaussian noise to edge logits"""
        noise = torch.randn_like(edge_logits) * self.noise_scale
        return edge_logits + noise

    def post_batch(
            self,
            model_manager: Any,
            batch: Any,
            **kwargs
    ) -> dict:
        node_emb = model_manager.gnn(batch.x, batch.edge_index)
        src_emb = node_emb[batch.edge_label_index[0]]
        dst_emb = node_emb[batch.edge_label_index[1]]

        if hasattr(model_manager.gnn, 'decode'):
            edge_logits = model_manager.gnn.decode(src_emb, dst_emb).squeeze(-1)
        else:
            edge_logits = (src_emb * dst_emb).sum(dim=-1)

        if self.noise_type == "reverse_sigmoid":
            modified_logits = self._apply_reverse_sigmoid_binary(edge_logits)
        elif self.noise_type == "random":
            modified_logits = self._apply_random_noise(edge_logits)
        else:
            modified_logits = edge_logits

        edge_labels = batch.edge_label.float()
        modified_loss = model_manager.loss_function(modified_logits, edge_labels)

        return {
            "outputs": modified_logits,
            "loss": modified_loss,
            "original_logits": edge_logits.detach()
        }