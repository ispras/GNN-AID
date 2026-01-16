from typing import Any, Optional, List, Callable, Union
from typing import Callable

import sklearn.metrics
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing


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
    for _, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            handlers.append(layer.register_message_forward_hook(attention_message_hook(att, layer)))
        elif isinstance(layer, torch.nn.Module):
            new_handlers = apply_attention_to_messages(layer, att)
            handlers.extend(new_handlers)
    return handlers


def attention_message_hook(
        att: Optional[torch.Tensor],
        layer: torch.nn.Module
):
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


class EdgeMaskingWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_edges: int):
        super().__init__()
        self.model = model
        self.edge_mask = nn.Parameter(torch.ones(num_edges))  # [E], requires_grad=True

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                if hasattr(module, 'add_self_loops'):
                    module.add_self_loops = False
                module.register_message_forward_hook(self._make_mask_hook())

    def _make_mask_hook(self):
        def hook(module, inputs, message_output):
            # message_output: [E, F]
            return message_output * self.edge_mask.to(message_output.device).view(-1, 1)
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Metric:
    available_metrics = {
        'Accuracy': sklearn.metrics.accuracy_score,
        'F1': sklearn.metrics.f1_score,
        'BalancedAccuracy': sklearn.metrics.balanced_accuracy_score,
        'Recall': sklearn.metrics.recall_score,
        'Precision': sklearn.metrics.precision_score,
        'Jaccard': sklearn.metrics.jaccard_score,
    }

    @staticmethod
    def add_custom(
            name: str,
            compute_function: Callable
    ) -> None:
        """
        Register a custom metric.
        Example for accuracy:

        >>> Metric.add_custom('accuracy', lambda y_true, y_pred, normalize=False:
        >>>     int((y_true == y_pred).sum()) / (len(y_true) if normalize else 1))

        :param name: name to refer to this metric
        :param compute_function: function which computes metric result:
         f(y_true, y_pred, **kwargs) -> value
        """
        if name in Metric.available_metrics:
            raise NameError(f"Metric '{name}' already registered, use another name")
        Metric.available_metrics[name] = compute_function

    def __init__(
            self,
            name: str,
            mask: Union[str, List[bool], torch.Tensor],
            **kwargs
    ):
        """
        :param name: name to refer to this metric
        :param mask: 'train', 'val', 'test', or a bool valued list
        :param kwargs: params used in compute function
        """
        self.name = name
        self.mask = mask
        self.kwargs = kwargs

    def compute(
            self,
            y_true,
            y_pred
    ):
        if self.name in Metric.available_metrics:
            if y_true.device != "cpu":
                y_true = y_true.cpu()
            if y_pred.device != "cpu":
                y_pred = y_pred.cpu()
            return Metric.available_metrics[self.name](y_true, y_pred, **self.kwargs)
        raise NotImplementedError()

    @staticmethod
    def create_mask_by_target_list(
            y_true,
            target_list: List = None
    ) -> torch.Tensor:
        if target_list is None:
            mask = [True] * len(y_true)
        else:
            mask = [False] * len(y_true)
        for i in target_list:
            if 0 <= i < len(mask):
                mask[i] = True
        return tensor(mask)


class GNNConstructorError(Exception):
    def __init__(
            self,
            *args
    ):
        self.message = args[0] if args else None

    def __str__(
            self
    ):
        if self.message:
            return f"GNNConstructorError: {self.message}"
        else:
            return "GNNConstructorError has been raised!"

