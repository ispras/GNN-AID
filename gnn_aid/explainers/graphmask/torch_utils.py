import math

import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
from torch.nn import Linear, LayerNorm, init

class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = self.clip(s)

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


class LagrangianOptimization:

    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    batch_size_multiplier = None
    update_counter = 0

    def __init__(self, original_optimizer, device='cpu', init_alpha=0.55, min_alpha=-2, max_alpha=30, alpha_optimizer_lr=1e-2, batch_size_multiplier=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.batch_size_multiplier = batch_size_multiplier
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer

    def update(self, f, g):
        """
        L(x, lambda) = f(x) + lambda g(x)

        :param f_function:
        :param g_function:
        :return:
        """

        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()

            self.update_counter += 1
        else:
            self.original_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        loss.backward()

        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                self.optimizer_alpha.step()
        else:
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()

        if self.alpha.item() < -2:
            self.alpha.data = torch.full_like(self.alpha.data, -2)
        elif self.alpha.item() > 30:
            self.alpha.data = torch.full_like(self.alpha.data, 30)


class MultipleInputsLayernormLinear(torch.nn.Module):

    """
    Properly applies layernorm to a list of inputs, allowing for separate rescaling of potentially unnormalized components.
    This is inspired by the implementation of layer norm for LSTM from the original paper.
    """

    components = None

    def __init__(self, input_dims, output_dim, init_type="xavier", force_output_dim=None, requires_grad=True):
        super(MultipleInputsLayernormLinear, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim if force_output_dim is None else force_output_dim
        self.init_type = init_type
        self.components = len(input_dims)

        self.transforms = []
        self.layer_norms = []
        for i, input_dim in enumerate(input_dims):
            self.transforms.append(Linear(input_dim, output_dim, bias=False))
            self.layer_norms.append(LayerNorm(output_dim))

        self.transforms = torch.nn.ModuleList(self.transforms)
        self.layer_norms = torch.nn.ModuleList(self.layer_norms)

        self.full_bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self):
        fan_in = sum(self.input_dims)

        std = math.sqrt(2.0 / float(fan_in + self.output_dim))
        a = math.sqrt(3.0) * std # Calculate uniform bounds from standard deviation

        for transform in self.transforms:
            if self.init_type == "xavier":
                init._no_grad_uniform_(transform.weight, -a, a)
            else:
                print("did not implement he init")
                exit()

        init.zeros_(self.full_bias)

        for layer_norm in self.layer_norms:
            layer_norm.reset_parameters()

    def forward(self, input_tensors):
        output = self.full_bias

        for component in range(self.components):
            tensor = input_tensors[component]
            transform = self.transforms[component]
            norm = self.layer_norms[component]

            partial = transform(tensor)
            result = norm(partial)
            output = output + result

        return output / self.components


class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(dim=-1)