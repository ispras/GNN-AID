from collections import namedtuple
from graphviz import Digraph
from torch.autograd import Variable
import torch

from torch import device
from torch.cuda import is_available
from torch_geometric.nn import MessagePassing

from data_structures.configs import ModelConfig
from models_builder.gnn_constructor import FrameworkGNNConstructor, GNNStructure
from datasets.datasets_manager import DatasetManager

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width) + 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


class NTensor:
    def __init__(self, tensor: torch.Tensor, name=None):
        self.t = tensor
        self.name = name if name else str(id(tensor))

    @property
    def id(self):
        return str(id(self.t))

    def __str__(self):
        return self.name + '\n----------------\n' + str(self.t)


class NOperation:
    def __init__(self, obj, **params):
        self.obj = obj
        self.params = {}
        self.params.update(params)

    @property
    def id(self):
        return self.obj if isinstance(self.obj, str) else str(id(self.obj))

    @property
    def name(self):
        return self.params.get("name", None)

    @property
    def fn(self):
        if 'fn' in self.params:
            return self.params["fn"]
        return None

    def __str__(self):
        res = 'OP\n'
        res += '\n'.join(f"{k}={v}" for k, v in self.params.items())
        if isinstance(self.obj, torch.Tensor):
            res += '\n------------\n' + str(self.obj)
        return res


class OperationalGraph:
    """ Tensors, operations and their full computational graph
    """

    def __init__(self, gnn: torch.nn.Module, *args, **kwargs):
        """
        :param gnn: GNN model, subclass of torch.nn.Module
        :param args: forward arguments
        :param kwargs: forward keyword arguments
        """
        self.gnn = gnn
        self.args = args
        self.kwargs = kwargs

        self.tensors = {}  # id -> tensor
        self.ops = {}  # id -> object
        self.edges = {}  # (id1, id2) -> attrs
        self.out_edges = {}  # id1 -> set of (id2)
        self.in_edges = {}  # id2 <- set of (id1)
        self.triples = []  # (operation, inputs, output)

        self._mess_out = {}  # module_id -> message out tensor

    def _add_tensor(self, obj, name=None):
        assert isinstance(obj, torch.Tensor)
        _id = str(id(obj))
        if _id in self.tensors:
            print(f"tensor {_id} already added")
            if name is None:
                name = self.tensors[_id].name
            elif self.tensors[_id].name:
                name += ',\n' + self.tensors[_id].name
        self.tensors[_id] = NTensor(obj, name=name)
        return self.tensors[_id]

    def _add_op(self, obj, **params):
        if isinstance(obj, str):
            _id = str(obj)
        else:
            _id = str(id(obj))
        if _id in self.ops:
            print(f"objects {_id} already added")
        self.ops[_id] = NOperation(obj, **params)

    def mess_hook(self, module, msg_kwargs, output):
        msg_kwargs = msg_kwargs[0]
        module_id = str(id(module))
        inputs = []
        for k, v in msg_kwargs.items():
            self.add_op(v, name='message.' + k)
            inputs.append(v)
        operation = module_id + '_mess'
        operation_id = operation
        self._add_op(operation, name='~mess')

        assert isinstance(output, torch.Tensor)
        self._add_tensor(output)
        self.add_edge(operation_id, str(id(output)), style='dotted')
        self._mess_out[module_id] = output

        for input in inputs:
            if isinstance(input, torch.Tensor):
                self._add_tensor(input)
                self.add_edge(str(id(input)), operation_id, style='dotted')

    def agg_hook(self, module, aggr_kwargs, output):
        aggr_kwargs = aggr_kwargs[0]
        module_id = str(id(module))
        operation = module_id + '_aggr'
        operation_id = operation
        self._add_op(operation, name='~aggr')

        assert isinstance(output, torch.Tensor)
        self._add_tensor(output)
        self.add_edge(operation_id, str(id(output)), style='dotted')

        input = self._mess_out[module_id]
        self._add_tensor(input)
        self.add_edge(str(id(input)), operation_id, style='dotted')

    def prop_hook(self, module, inputs, output):
        module_id = str(id(module))
        edge_index, size, kwargs = inputs
        self._add_tensor(edge_index, name="edge_index'")
        mess_id = module_id + '_mess'
        self.add_edge(str(id(edge_index)), mess_id)

        # output == agg.output ? FIXME Misha

    def hook(self, module, input, output):
        module_id = str(id(module))
        operation = module_id + '_' + str(module)
        operation_id = operation
        self._add_op(operation, name=f"~{str(module)}", fn=module)

        for ix, inp in enumerate(input):
            self._add_tensor(inp, name=f"hook-input-{str(module)}-{ix}")
            self.add_edge(str(id(inp)), operation_id, style='dotted')
        self._add_tensor(output, name=f"hook-output-{str(module)}")
        self.add_edge(operation_id, str(id(output)), style='dotted')

    def my_hook(self, module, ix, feat, edge_index, input, middle, output):
        module_id = str(id(module))
        self._add_tensor(feat, name="INPUT FEATURES")
        self._add_tensor(input, name=f"my-hook[{ix}]-input-{str(module)}")
        self._add_tensor(middle, name=f"my-hook[{ix}]-middle-{str(module)}")
        self._add_tensor(output, name=f"my-hook[{ix}]-output-{str(module)}")
        self._add_tensor(edge_index, name="edge_index")

        activation = str(id(module.activations[ix])) + str(ix) + '_activation'
        activation_id = activation
        self._add_op(activation, name=f'~activation[{ix}]', fn=module.activations[ix])

        assert isinstance(output, torch.Tensor)
        self._add_tensor(output)
        self.add_edge(str(id(middle)), activation_id, style='dotted')
        self.add_edge(activation_id, str(id(output)), style='dotted')

    def add_op(self, op, **params):
        if isinstance(op, torch.Tensor):
            self._add_tensor(op, name=params.get("name", None))
        else:
            self._add_op(op, **params)
            op_id = str(id(op))

            attrs = dict()
            for attr in dir(op):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(op, attr)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    attrs[attr] = "[saved tensor]"
                    if attr != "result":
                        self._add_tensor(val, name=attr)
                        self.add_edge(str(id(val)), op_id)
                elif isinstance(val, tuple):
                    for ix, t in enumerate(val):
                        if torch.is_tensor(t):
                            attrs[attr] = "[saved tensors]"
                            self._add_tensor(t, name=attr + ix)
                            # self.add_edge(str(id(t)), op_id, dir="none")
                else:
                    attrs[attr] = str(val)

    def add_edge(self, id1, id2, **attrs):
        if (id1, id2) in self.edges:
            print(f"Edge {(id1, id2)} already added")
            return
        self.edges[(id1, id2)] = attrs
        if id1 not in self.out_edges:
            self.out_edges[id1] = {id2}
        else:
            self.out_edges[id1].add(id2)
        if id2 not in self.in_edges:
            self.in_edges[id2] = {id1}
        else:
            self.in_edges[id2].add(id1)

    def rm_edge(self, id1, id2):
        del self.edges[(id1, id2)]
        self.out_edges[id1].remove(id2)
        self.in_edges[id2].remove(id1)

    def print(self):
        print(f"Tensors: ({len(self.tensors)})")
        for _id, t in self.tensors.items():
            print(_id, t)
        print(f"Ops: ({len(self.ops)})")
        for _id, t in self.ops.items():
            print(_id, t)
        print(f"Triples: ({len(self.triples)})")
        for tr in self.triples:
            print(tr)

    def finalize(self):
        """ Match tensors and ops
        """
        # Match tensors with grad_fn equal to some existing op
        for _id, t in self.tensors.items():
            if hasattr(t.t, "grad_fn"):
                fn_id = str(id(t.t.grad_fn))
                if fn_id in self.ops:
                    print("in")
                    self.add_edge(fn_id, t.id)
                    # If op1 -> tensor and op1 -> op2 then op1 -> tensor -> op2
                    outs = list(self.out_edges[fn_id])
                    for out in outs:
                        if out in self.ops:
                            self.add_edge(t.id, out)
                            self.rm_edge(fn_id, out)

        # remove AccumulatedGrad
        to_remove = set()
        for _id, op in self.ops.items():
            if op.name == "AccumulateGrad":
                ins = list(self.in_edges[_id])
                outs = list(self.out_edges[_id])
                assert len(ins) == len(outs) == 1
                attr = self.edges[(ins[0], _id)]
                self.add_edge(ins[0], outs[0], **attr)
                self.rm_edge(ins[0], _id)
                self.rm_edge(_id, outs[0])
                to_remove.add(_id)
        for _id in to_remove:
            del self.ops[_id]

    # def light_graph(self):
    #     """
    #
    #     """
    #     res = {
    #         "X": None,
    #         "M": [],
    #         "Agg": [],
    #         "Act": [],
    #     }
    #
    #     name_node = {nt.name: nt for nt in self.tensors.values()}
    #     name_op = {nop.name: nop for nop in self.ops.values()}
    #
    #     # Light graph
    #     tensors = []
    #     ops = []
    #     edges = []
    #
    #     # Get X
    #     for name, node in name_node.items():
    #         if "INPUT FEATURES" in name:
    #             res["X"] = node.t
    #             tensors.append(node)
    #             break
    #
    #     # Get messages
    #     for name, op in name_op.items():
    #         if name == "~mess":
    #             _id = op.id
    #             ops.append(op)
    #             ins = self.in_edges[_id]
    #             m_in = None
    #             for node_id in ins:
    #                 node = self.tensors[node_id]
    #                 if node.name == "message.x_j":
    #                     m_in = node.t
    #                     tensors.append(node)
    #                     edges.append((node, op))
    #                     break
    #             assert m_in is not None
    #
    #             outs = self.out_edges[_id]
    #             assert len(outs) == 1
    #             out = self.tensors[list(outs)[0]]
    #             m_out = out.t
    #             tensors.append(out)
    #             edges.append((op, out))
    #
    #             res["M"].append((m_in, m_out))
    #
    #     # Get aggregations
    #     for name, op in name_op.items():
    #         if name == "~aggr":
    #             _id = op.id
    #             ops.append(op)
    #             ins = self.in_edges[_id]
    #             assert len(ins) == 1
    #             _in = self.tensors[list(ins)[0]]
    #             agg_in = _in.t
    #             tensors.append(_in)
    #             edges.append((_in, op))
    #
    #             outs = self.out_edges[_id]
    #             assert len(outs) == 1
    #             out = self.tensors[list(outs)[0]]
    #             agg_out = out.t
    #             tensors.append(out)
    #             edges.append((op, out))
    #
    #             res["Agg"].append((agg_in, agg_out))
    #
    #     # Get activations
    #     for name, op in name_op.items():
    #         if name == "~activation":
    #             _id = op.id
    #             ops.append(op)
    #             ins = self.in_edges[_id]
    #             assert len(ins) == 1
    #             _in = self.tensors[list(ins)[0]]
    #             agg_in = _in.t
    #             tensors.append(_in)
    #             edges.append((_in, op))
    #
    #             outs = self.out_edges[_id]
    #             assert len(outs) == 1
    #             out = self.tensors[list(outs)[0]]
    #             agg_out = out.t
    #             tensors.append(out)
    #             edges.append((op, out))
    #
    #             res["Act"].append((agg_in, agg_out))
    #
    #     # Add model params
    #     # TODO Misha
    #
    #     tensors = {t.id: t for t in tensors}
    #     ops = {t.id: t for t in ops}
    #     edges = {(x.id, y.id): {} for x, y in edges}
    #     self.render_graph(tensors, ops, edges)

    def render_graph(self, tensors=None, ops=None, edges=None):
        tensor_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='10',
            ranksep='0.1',
            height='0.2',
            # fillcolor='lightgreen',
            fontname='monospace')
        op_attr = dict(style='filled',
                       shape='ellipse',
                       fontsize='10',
                       # ranksep='0.1',
                       # height='0.2',
                       fontname='monospace')
        dot = Digraph(graph_attr=dict(size="12,12"))
        for _id, t in (tensors or self.tensors).items():
            dot.node(_id.replace(':', ''), str(t), **tensor_attr,
                     fillcolor='orange' if t.t.requires_grad else 'lightgreen')

        for _id, op in (ops or self.ops).items():
            dot.node(_id.replace(':', ''), str(op), **op_attr,
                     fillcolor='white' if op.name.startswith('~') else 'grey')

        for (id1, id2), attrs in (edges or self.edges).items():
            dot.edge(id1.replace(':', ''), id2.replace(':', ''), **attrs)

        dot.render("opGraph", format='pdf')
        dot.render("opGraph", format='svg')

    def make_dot(self, var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
        """ Produces Graphviz representation of PyTorch autograd graph.

        If a node represents a backward function, it is gray. Otherwise, the node
        represents a tensor and is either blue, orange, or green:
         - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
             fields will be populated during `.backward()`)
         - Orange: saved tensors of custom autograd functions as well as those
             saved by built-in backward nodes
         - Green: tensor passed in as outputs
         - Dark green: if any output is a view, we represent its base tensor with
             a dark green node.

        Args:
            var: output tensor
            params: dict of (name, tensor) to add names to node that requires grad
            show_attrs: whether to display non-tensor attributes of backward nodes
                (Requires PyTorch version >= 1.9)
            show_saved: whether to display saved tensor nodes that are not by custom
                autograd functions. Saved tensor nodes for custom functions, if
                present, are always displayed. (Requires PyTorch version >= 1.9)
            max_attr_chars: if show_attrs is `True`, sets max number of characters
                to display for any given attribute.
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}
        else:
            param_map = {}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='10',
                         ranksep='0.1',
                         height='0.2',
                         fontname='monospace')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        # cg = ComputationGraph()
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def get_var_name(var, name=None):
            if not name:
                name = param_map[id(var)] if id(var) in param_map else ''
            return '%s\n %s' % (name, size_to_str(var.size()))

        def add_nodes(fn):
            assert not torch.is_tensor(fn)
            if fn in seen:
                return
            seen.add(fn)

            if show_saved:
                for attr in dir(fn):
                    if not attr.startswith(SAVED_PREFIX):
                        continue
                    val = getattr(fn, attr)
                    seen.add(val)
                    attr = attr[len(SAVED_PREFIX):]
                    if torch.is_tensor(val):
                        dot.edge(str(id(fn)), str(id(val)), dir="none")
                        dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                    if isinstance(val, tuple):
                        for i, t in enumerate(val):
                            if torch.is_tensor(t):
                                name = attr + '[%s]' % str(i)
                                dot.edge(str(id(fn)), str(id(t)), dir="none")
                                dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')

            if hasattr(fn, 'variable'):
                # if grad_accumulator, add the node for `.variable`
                var = fn.variable
                seen.add(var)
                dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
                self.add_op(var, name=get_var_name(var))
                dot.edge(str(id(var)), str(id(fn)))
                self.add_edge(str(id(var)), str(id(fn)))
                # cg.add_edge(str(id(var)), str(id(fn)))

            # add the node for this grad_fn
            dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))
            self.add_op(fn, name=get_fn_name(fn, show_attrs, max_attr_chars))

            # recurse
            if hasattr(fn, 'next_functions'):
                for u in fn.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(fn)))
                        self.add_edge(str(id(u[0])), str(id(fn)))
                        # cg.add_edge(str(id(u[0])), str(id(fn)))
                        add_nodes(u[0])

            # note: this used to show .saved_tensors in pytorch0.2, but stopped
            # working* as it was moved to ATen and Variable-Tensor merged
            # also note that this still works for custom autograd functions
            if hasattr(fn, 'saved_tensors'):
                for t in fn.saved_tensors:
                    dot.edge(str(id(t)), str(id(fn)))
                    self.add_edge(str(id(t)), str(id(fn)))
                    # cg.add_edge(str(id(t)), str(id(fn)))
                    dot.node(str(id(t)), get_var_name(t), fillcolor='orange')
                    self.add_op(t, name=get_var_name(t))

        def add_base_tensor(var, color='darkolivegreen1'):
            if var in seen:
                return
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor=color)
            self.add_op(var, name=get_var_name(var))
            if var.grad_fn:
                add_nodes(var.grad_fn)
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                self.add_edge(str(id(var.grad_fn)), str(id(var)))
                # cg.add_edge(str(id(var.grad_fn)), str(id(var)))
            if var._is_view():
                add_base_tensor(var._base, color='darkolivegreen3')
                dot.edge(str(id(var._base)), str(id(var)), style="dotted")
                # self.add_edge(str(id(var._base)), str(id(var)), style="dotted")
                # cg.add_edge(str(id(var._base)), str(id(var)), style="dotted")

        # handle multiple outputs
        if isinstance(var, tuple):
            for v in var:
                add_base_tensor(v)
        else:
            add_base_tensor(var)

        return dot

    def full_forward_graph(self):
        """ Build a full computation graph
        """
        for module in gnn.modules():
            if isinstance(module, FrameworkGNNConstructor):
                module.register_my_forward_hook(self.my_hook)
            elif isinstance(module, MessagePassing):
                module.register_message_forward_hook(self.mess_hook)
                module.register_aggregate_forward_hook(self.agg_hook)
                module.register_propagate_forward_hook(self.prop_hook)
                module.register_forward_hook(self.hook)
            else:
                module.register_forward_hook(self.hook)

        y = gnn(*self.args, **self.kwargs)
        self.make_dot(y, params=dict(gnn.named_parameters()), show_attrs=True, show_saved=True)
        self.finalize()
        self.print()
        self.render_graph()

    def light_forward_graph(self, render=False):
        """ Build a light version of computation graph
        """
        assert isinstance(gnn, FrameworkGNNConstructor)
        for module in gnn.modules():
            if isinstance(module, GNNStructure):
                module.register_my_forward_hook(self.my_hook)
            else:
                module.register_forward_hook(self.hook)

        # Forward call when all hooks activate
        y = gnn(*self.args, **self.kwargs)

        inputs = []
        for i, x in enumerate((*self.args, *self.kwargs.values())):
            if isinstance(x, torch.Tensor):
                nt = self._add_tensor(x, name=f'INPUT[{i}]')
                inputs.append(nt.id)
        nt = self._add_tensor(y, name='RESULT')
        result = nt.id
        inputs = set(inputs)

        # Filter only part between inputs and result
        nodes = set(inputs)

        def walk_back(_id, tail):
            if _id in inputs:
                nodes.update(tail)
                return
            if _id not in self.in_edges:
                return
            ins = self.in_edges[_id]
            if len(ins) == 0:
                return
            elif len(ins) == 1:
                walk_back(next(iter(ins)), tail + [_id])
            else:
                for _in in ins:
                    walk_back(_in, list(tail) + [_id])

        walk_back(result, [])

        tensors = {id: _ for id, _ in self.tensors.items() if id in nodes}
        ops = {id: _ for id, _ in self.ops.items() if id in nodes}
        edges = {(id1, id2): _ for (id1, id2), _ in self.edges.items() if id1 in nodes and id2 in nodes}
        if render:
            self.render_graph(tensors=tensors, ops=ops, edges=edges)

        # self.render_graph()
        return inputs, result, tensors, ops, edges


def computation_graph_data(x, edge_index, inputs, result, tensors, ops, edges):
    """ Create data structure describing light computation graph
    """
    assert str(id(x)) in inputs
    assert str(id(edge_index)) in inputs

    layers = []
    out_edges = {id1: id2 for (id1, id2), _ in edges.items()}  # id1 -> id2

    # Start from x, each op extending MessagePassing means the next layer
    current_id = str(id(x))
    layer = {}
    layer["tensors"] = []
    layer["ops"] = []
    layer["tensors"].append(tensors[current_id])
    while True:
        if current_id not in out_edges: break
        next_id = out_edges[current_id]
        if next_id in ops:
            op = ops[next_id]
            if isinstance(op.fn, MessagePassing):
                layer["conv"] = op
                layers.append(layer)
                layer = {}
                layer["tensors"] = []
                layer["ops"] = []
            else:
                layer["ops"].append(op)

        elif next_id in tensors:
            layer["tensors"].append(tensors[next_id])
        current_id = next_id

    return layers


if __name__ == '__main__':
    my_device = device('cuda' if is_available() else 'cpu')
    torch.random.manual_seed(0)

    # single
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=("single-graph", "custom", 'example'),
        features=FeatureConfig(node_attr=['a']),
        labeling='binary',
        dataset_ver_ind=0)

    # # multi
    # dataset, data, results_dataset_path = DatasetManager.get_pytorch_geometric(
    #     full_name=("multiple-graphs", "custom", 'example'),
    #     features=FeatureConfig(node_attr=['type']),
    #     labeling='binary',
    #     dataset_ver_ind=0)

    sage1: torch.nn.Module = GNNStructure(
        # conv_classes=('SGConv', 'SGConv'),
        conv_classes=('SAGEConv',),
        layers_sizes=(dataset.num_node_features, dataset.num_classes),
        activations=('torch.nn.functional.log_softmax',),
    )
    # sage2: torch.nn.Module = GNNStructure(
    #     # conv_classes=('SGConv', 'SGConv'),
    #     conv_classes=('SAGEConv','SAGEConv',),
    #     layers_sizes=(dataset.num_node_features, 3, dataset.num_classes),
    #     activations=('torch.relu', 'torch.nn.functional.log_softmax',),
    # )
    # sage3: torch.nn.Module = GNNStructure(
    #     # conv_classes=('SGConv', 'SGConv'),
    #     conv_classes=('SAGEConv', 'SAGEConv', 'SAGEConv',),
    #     layers_sizes=(dataset.num_node_features, 3, 4, dataset.num_classes),
    #     activations=('torch.relu', 'torch.relu', 'torch.nn.functional.log_softmax',),
    # )
    # gcn1: torch.nn.Module = GNNStructure(
    #     conv_classes=('GCNConv', ),
    #     layers_sizes=(dataset.num_node_features, dataset.num_classes),
    #     activations=('torch.nn.functional.log_softmax',),
    # )
    # gcn2: torch.nn.Module = GNNStructure(
    #     conv_classes=('GCNConv', 'GCNConv'),
    #     layers_sizes=(dataset.num_node_features, 3, dataset.num_classes),
    #     activations=('torch.relu', 'torch.nn.functional.log_softmax',),
    # )
    # gat1: torch.nn.Module = GNNStructure(
    #     conv_classes=('GATConv', ),
    #     layers_sizes=(dataset.num_node_features, dataset.num_classes),
    #     activations=('torch.nn.functional.log_softmax',),
    # )
    # gat2: torch.nn.Module = GNNStructure(
    #     conv_classes=('GATConv', 'GATConv'),
    #     layers_sizes=(dataset.num_node_features, 3, dataset.num_classes),
    #     activations=('torch.relu', 'torch.nn.functional.log_softmax',),
    # )

    sage = FrameworkGNNConstructor(ModelConfig(
        structure=[
            {
                'label': 'n',
                'conv': {
                    'conv_name': 'SAGEConv',
                    'conv_kwargs': {
                        'in_channels': dataset.num_node_features,
                        'out_channels': dataset.num_classes,
                    },
                },
                'activation': {
                    'activation_name': 'LogSoftmax',
                    'activation_kwargs': None,
                },
            },

        ])
    )

    gat_gin_lin = FrameworkGNNConstructor(ModelConfig(
        structure=[
            {
                'label': 'n',
                'gin': [
                    {
                        'lin': {
                            'lin_name': 'Linear',
                            'lin_kwargs': {
                                'in_features': dataset.num_node_features,
                                'out_features': 16,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 16,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'lin': {
                            'lin_name': 'Linear',
                            'lin_kwargs': {
                                'in_features': 16,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                ],
                'gin_kwargs': None,
                'connections': [
                    {
                        'into_layer': 1,
                        'connection_kwargs': {
                            'pool': {
                                'pool_type': 'global_add_pool',
                            },
                            'aggregation_type': 'cat',
                        },
                    },
                ],
            },

            {
                'label': 'g',
                'lin': {
                    'lin_name': 'Linear',
                    'lin_kwargs': {
                        'in_features': 16,
                        'out_features': dataset.num_classes,
                    },
                },
                'activation': {
                    'activation_name': 'Softmax',
                    'activation_kwargs': None,
                },
            },
        ])
    )

    gnn = sage
    print(gnn)

    og = OperationalGraph(gnn, data.x, data.edge_index)
    og.full_forward_graph()
    # _ = og.light_forward_graph()
    # layers = computation_graph_data(data.x, data.edge_index, *_)

    # og.print()
