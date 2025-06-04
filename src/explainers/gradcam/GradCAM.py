from typing import Any, Callable, List, Tuple, Union, Dict

import captum.attr as ca
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.conv.gcn_conv
from captum.attr._utils.common import (
    _format_additional_forward_args,
    _format_attributions,
    _format_input,
)
from captum.attr._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum.attr._utils.typing import (
    TargetType,
)
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils.loop import add_remaining_self_loops

from dig.xgraph.method.base_explainer import WalkBase
from dig.xgraph.models.utils import subgraph, normalize

from explainers.explainer_results_to_json import SubgraphExplanation

EPS = 1e-15


class GradCAM(WalkBase):
    r"""
    An implementation of GradCAM on graph in
    `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)

    .. note::
        For node classification model, the :attr:`explain_graph` flag is False.
        For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self, model: nn.Module, explain_graph: bool = False):
        super().__init__(model, explain_graph=explain_graph)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) \
            -> Union[Tuple[None, List, List[Dict]], Tuple[List, List, List[Dict]]]:
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype: (None, list, list)

        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        self.model.eval()
        super().forward(x, edge_index)

        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            hard_edge_masks = [self.control_sparsity(mask, kwargs.get('sparsity')).sigmoid() for mask in edge_masks]
        else:
            # --- setting GradCAM ---
            # class model_node(nn.Module):
            #     def __init__(self, cls):
            #         super().__init__()
            #         self.cls = cls
            #         print(cls)
            #         print(type(cls))
            #         self.convs = cls.model.convs
            #
            #     def forward(self, *args, **kwargs):
            #         return self.cls.model(*args, **kwargs)[node_idx]

            if self.explain_graph:
                model = self.model
            else:
                # model = model_node(self)
                pass
            # self.explain_method = GraphLayerGradCam(model, model.convs[-1])
            # нет атрибута .convs т.к. он юзается в class GCNNet(GNNBase) из DIG/benchmarks/xgraph/gnnNets.py
            # поэтому:
            # ((list(self.model._modules.items())[-1])[-1])

            mod = []
            for m in self.model.modules():
                mod.append(m)
            self.explain_method = GraphLayerGradCam(self.model, mod[4])
            # где list(model._modules.items())[-1][-1] возвращает последний сверточный слой <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>
            # --- setting end ---

            edge_masks = []
            hard_edge_masks = []
            for ex_label in ex_labels:
                attr_wo_relu = self.explain_method.attribute(x, ex_label, additional_forward_args=edge_index)
                mask = normalize(attr_wo_relu.relu())
                mask = mask.squeeze()
                mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
                edge_masks.append(mask.detach())
                mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                mask = mask.sigmoid()
                hard_edge_masks.append(mask.detach())

        # Store related predictions for further evaluation.
        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        return edge_masks, hard_edge_masks, related_preds



class GraphLayerGradCam(ca.LayerGradCam):

    def __init__(
            self,
            forward_func: Callable,
            layer: Module,
            device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target: TargetType = None,
            additional_forward_args: Any = None,
            attribute_to_layer_input: bool = False,
            relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            relu_attributions (bool, optional): Indicates whether to
                        apply a ReLU operation on the final attribution,
                        returning only non-negative attributions. Setting this
                        flag to True matches the original GradCAM algorithm,
                        otherwise, by default, both positive and negative
                        attributions are returned.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
        Examples::

            # >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            # >>> # and returns an Nx10 tensor of class probabilities.
            # >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            # >>> # and the output of this layer has dimensions Nx50x8x8.
            # >>> # It is the last convolution layer, which is the recommended
            # >>> # use case for GradCAM.
            # >>> net = ImageClassifier()
            # >>> layer_gc = LayerGradCam(net, net.conv4)
            # >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            # >>> # Computes layer GradCAM for class 3.
            # >>> # attribution size matches layer output except for dimension
            # >>> # 1, so dimensions of attr would be Nx1x8x8.
            # >>> attr = layer_gc.attribute(input, 3)
            # >>> # GradCAM attributions are often upsampled and viewed as a
            # >>> # mask to the input, since the convolutional layer output
            # >>> # spatially matches the original input image.
            # >>> # This can be done with LayerAttribution's interpolate method.
            # >>> upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))
        """
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals, is_layer_tuple = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        # Gradient Calculation end

        # Addition: shape from PyG to General PyTorch
        layer_gradients = tuple(layer_grad.transpose(0, 1).unsqueeze(0)
                                for layer_grad in layer_gradients)

        layer_evals = tuple(layer_eval.transpose(0, 1).unsqueeze(0)
                            for layer_eval in layer_evals)
        # end

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        # what I add: shape from General PyTorch to PyG

        scaled_acts = tuple(scaled_act.squeeze(0).transpose(0, 1)
                            for scaled_act in scaled_acts)

        # end

        return _format_attributions(is_layer_tuple, scaled_acts)


class GradCAMOut(SubgraphExplanation):
    def __init__(self, edge_mask, node_mask, node_feat_mask, data):
        super().__init__(edge_mask, node_mask, node_feat_mask, data)

    def add_edges(self):
        dict_imp_edges = {}
        tensor_transpose = list(torch.transpose(self.dataset.data.edge_index, 0, 1))
        if self.edge_mask is not None:
            _, __, dict_imp_edges, ___ = self.get_important_edges()
            keys = dict_imp_edges.keys()

            for i in range(len(tensor_transpose)):
                key = str(tuple(map(int, tensor_transpose[i])))
                if key not in keys:
                    dict_imp_edges[key] = None
        else:
            for i in range(len(tensor_transpose)):
                key = str(tuple(map(int, tensor_transpose[i])))
                dict_imp_edges[key] = None
        self.dictionary['E'] = dict_imp_edges

    def add_features(self):
        dict_features = {}
        if self.node_feat_mask is not None:
            features = list(self.node_feat_mask)
            for i in range(len(features)):
                dict_features[str(i)] = format(float(features[i]), '.4f')
        else:
            for i in range(self.dataset.data.x.size(1)):
                dict_features[str(i)] = None
        self.dictionary['F'] = dict_features

    def add_nodes(self):
        dict_imp_nodes = {}
        if self.node_mask is not None:
            _, __, ___, dict_imp_nodes = self.get_important_edges()
            keys = dict_imp_nodes.keys()
            for i in range(self.dataset.data.x.size(0)):
                key = str(i)
                if key not in keys:
                    dict_imp_nodes[key] = None
        else:
            for i in range(self.dataset.data.x.size(0)):
                dict_imp_nodes[str(i)] = None
        self.dictionary['V'] = dict_imp_nodes

    def get_features(self):
        dict_features = {}
        features = list(self.node_feat_mask)
        if self.node_feat_mask is not None:
            for i in range(len(features)):
                dict_features[str(i)] = format(float(features[i]), '.4f')
            return dict_features

    def get_important_edges(self):
        edge_mask = list(map(float, self.edge_mask))
        tensor_transpose = torch.transpose(self.dataset.data.edge_index, 0, 1)

        notnull_positions = []
        for i in range(len(edge_mask)):
            if edge_mask[i] != 0:
                notnull_positions.append(i)
        print(notnull_positions)

        important_edges = []
        dict_important_edges = {}
        important_nodes = []
        dict_important_nodes = {}
        for elem in notnull_positions:
            print(elem)
            if elem < len(list(tensor_transpose)):
                important_edges.append(list(map(int, tensor_transpose[elem])))
                dict_important_edges[str(tuple(important_edges[-1]))] = format(float(edge_mask[elem]), '.4f')
            else:
                important_nodes.append(int(elem) - len(tensor_transpose))
                dict_important_nodes[str(important_nodes[-1])] = format(float(edge_mask[elem]), '.4f')
        dict_important_edges = dict(sorted(dict_important_edges.items(), key=lambda x: x[1], reverse=True))
        print(dict_important_edges)
        dict_important_nodes = dict(sorted(dict_important_nodes.items(), key=lambda x: x[1], reverse=True))
        return important_edges, important_nodes, dict_important_edges, dict_important_nodes