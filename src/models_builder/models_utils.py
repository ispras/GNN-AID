from typing import Any, Union, Optional, List
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size

from attacks.metattack.utils import add_self_loops


def apply_message_gradient_capture(
        layer: Any,
        name: str
) -> None:
    """
    # Example how get Tensors
    # for name, layer in self.gnn.named_children():
    #     if isinstance(layer, MessagePassing):
    #         print(f"{name}: {layer.get_message_gradients()}")
    """
    original_message = layer.message
    layer.message_gradients = {}

    def capture_message_gradients(
            x_j: torch.Tensor,
            *args,
            **kwargs
    ):
        x_j = x_j.requires_grad_()
        if layer.training:
            return original_message(x_j=x_j, *args, **kwargs)

        def save_message_grad(
                grad: torch.Tensor
        ) -> None:
            layer.message_gradients[name] = grad.detach()

        x_j.register_hook(save_message_grad)
        return original_message(x_j=x_j, *args, **kwargs)

    layer.message = capture_message_gradients

    def get_message_gradients(
    ) -> dict:
        return layer.message_gradients

    layer.get_message_gradients = get_message_gradients


# def apply_attention(
#         layer: Any,
#         name: str
# ) -> None:
#     """Modifies the forward method of the given layer to include edge_atten handling."""
#     original_forward = layer.forward
#
#     def modified_forward(self: Any, *args, edge_atten: Optional[Tensor] = None, **kwargs) -> Tensor:
#         # Inject edge_atten into kwargs if it's provided
#         if edge_atten is not None:
#             kwargs['edge_atten'] = edge_atten
#
#         return original_forward(*args, **kwargs)
#
#     layer.forward = modified_forward.__get__(layer)


def apply_decorator_to_graph_layers(
        model: Any,
        dec_f: Callable = apply_message_gradient_capture
) -> None:
    # TODO Kirill add more options
    """
    Example how use this def
    apply_decorator_to_graph_layers(gnn)
    """
    for name, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            dec_f(layer, name)
        elif isinstance(layer, torch.nn.Module):
            apply_decorator_to_graph_layers(layer, dec_f)


def apply_attention_to_messages(
        model: Any,
        att: torch.Tensor
) -> List[RemovableHandle]:
    handlers = []
    for name, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            handlers.append(layer.register_message_forward_hook(attention_message_hook(att, layer)))
        elif isinstance(layer, torch.nn.Module):
            new_handlers = apply_attention_to_messages(layer, att)
            handlers.extend(new_handlers)
    return handlers


def attention_message_hook(att, layer):
    if att is None:
        return lambda module, input, out: out
    else:
        if not hasattr(layer, 'add_self_loops') or not layer.add_self_loops:
            return lambda module, input, out: out * att[out.shape[0], :]  # TODO assert here?
        else:
            if hasattr(layer, 'heads'):
                return lambda module, input, out: out * att.view(att.shape[0], 1, 1)
            else:
                return lambda module, input, out: out * att
