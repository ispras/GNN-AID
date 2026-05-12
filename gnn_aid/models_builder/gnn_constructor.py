import copy
import json
import warnings
from collections import OrderedDict
from typing import Dict, Callable, Union, Iterator, Type, Tuple
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils import hooks
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing

from gnn_aid.aux.utils import import_by_name, CUSTOM_LAYERS_INFO_PATH, MODULES_PARAMETERS_PATH, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY
from gnn_aid.data_structures.configs import ModelConfig, ModelStructureConfig
from gnn_aid.data_structures.gen_config import CONFIG_CLASS_NAME, ConfigPattern
from .models_utils import GNNConstructorError


class GNNConstructor:
    """
    The base class of all models. Contains the following methods:
    forward, get_all_layer_embeddings, get_architecture, get_num_hops,
    reset_parameters, get_predictions, get_answer, get_name
    """

    def __init__(self,
                 model_config: ModelConfig = None,
                 ):
        """
        Args:
            model_config (ModelConfig): Model configuration. Defaults to an empty ModelConfig
                if not provided. Default value: `None`.
        """
        if model_config is None:
            # raise RuntimeError("model manager config must be specified")
            model_config = ModelConfig()
        self.obj_name = None
        self.model_config = model_config

    def forward(
            self
    ):
        raise NotImplementedError("forward can't be called, because it is not implemented")

    def decode(
            self
    ):
        """
        Compute an unnormalized score for an edge given embeddings of its two endpoint nodes.

        Returns:
            Unnormalized edge score.
        """
        raise NotImplementedError("decode can't be called, because it is not implemented")

    def get_all_layer_embeddings(
            self
    ):
        """
        Return the output embedding of each layer in the model.

        Returns:
            Vectors representing the input data from the output of each layer.
        """
        raise NotImplementedError("get_all_layer_embeddings can't be called, because it is not implemented")

    def get_architecture(
            self
    ):
        """
        Return the model architecture description for display in the frontend.

        Returns:
            Model architecture object (e.g. ModelStructureConfig or dict).
        """
        raise NotImplementedError("get_architecture can't be called, because it is not implemented")

    def get_num_hops(
            self
    ):
        """
        Return the number of graph convolution layers (hops).

        Required for some model interpretation algorithms.

        Returns:
            Number of message-passing hops.
        """
        raise NotImplementedError("get_num_hops can't be called, because it is not implemented")

    def reset_parameters(
            self
    ):
        """ Reset all model parameters. Required for the reset button in the frontend.
        """
        raise NotImplementedError("reset_parameters can't be called, because it is not implemented")

    def get_predictions(
            self
    ):
        """
        Return a probability distribution over classes for the input data.

        Required for some interpretation algorithms. Does not require overriding forward.

        Returns:
            Class probability vector.
        """
        raise NotImplementedError("get_predictions can't be called, because it is not implemented")

    def get_parameters(
            self
    ):
        """
        Return the model parameter iterator or matrix.

        Returns:
            Model parameters.
        """
        raise NotImplementedError("get_predictions can't be called, because it is not implemented")

    def get_answer(
            self
    ):
        """
        Return the hard class assignment for the input.

        Required for some interpretation methods. Does not require overriding forward or get_predictions.

        Returns:
            Predicted class index or label for the input.
        """
        raise NotImplementedError("get_answer can't be called, because it is not implemented")

    def get_name(
            self,
            obj_name_flag: bool = False,
            **kwargs
    ):
        """
        Serialize the model config and class name to a JSON string used as a unique identifier.

        Args:
            obj_name_flag (bool): If True, include the obj_name field. Default value: `False`.
            **kwargs: Extra key-value pairs merged into the name dict.

        Returns:
            JSON string uniquely identifying this model.
        """
        gnn_name = self.model_config.to_savable_dict().copy()
        gnn_name[CONFIG_CLASS_NAME] = self.__class__.__name__
        if obj_name_flag:
            gnn_name['obj_name'] = self.obj_name
        for key, value in kwargs.items():
            gnn_name[key] = value
        gnn_name = dict(sorted(gnn_name.items()))
        json_str = json.dumps(gnn_name, indent=2)
        return json_str

    def suitable_model_managers(
            self
    ):
        """
        Return a set of model manager class names compatible with this model.

        Model manager classes must inherit from GNNModelManager.

        Returns:
            Set of suitable model manager class name strings.
        """
        raise NotImplementedError("suitable_model_managers can't be called, because it is not implemented")

    # === Permanent methods (not to be overwritten)

    def get_hash(
            self
    ) -> str:
        """
        Compute the SHA-256 hash of the model name string used for storage paths.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        gnn_name = self.get_name()
        json_object = json.dumps(gnn_name)
        gnn_name_hash = hash_data_sha256(json_object.encode('utf-8'))
        return gnn_name_hash

    def get_full_info(
            self,
            tensor_size_limit: int = None
    ) -> dict:
        """
        Get available info about the model for the frontend.

        Args:
            tensor_size_limit (int): Maximum number of elements before a weight tensor is
                returned as a shape string instead of values. Default value: `None`.

        Returns:
            Dict with available keys from 'architecture', 'weights', 'neurons'.
        """
        # FIXMe architecture and weights can be not accessible
        result = {}
        try:
            # TODO use dataclass for ModelConfigStructure to make it serializable
            result["architecture"] = self.get_architecture().to_dict()
        except (AttributeError, NotImplementedError):
            pass
        try:
            result["weights"] = self.get_weights(tensor_size_limit=tensor_size_limit)
        except (AttributeError, NotImplementedError):
            pass
        try:
            result["neurons"] = self.get_neurons()
        except (AttributeError, NotImplementedError):
            pass

        return result

    def get_weights(
            self,
            tensor_size_limit
    ) -> dict:
        pass

    def get_neurons(
            self
    ) -> list:
        pass


class GNNConstructorTorch(
    GNNConstructor,
    torch.nn.Module
):
    """
    Base class for writing models using the torch library. Inherited from GNNConstructor and torch.nn.Module classes.
    """

    def __init__(
            self
    ):
        super().__init__()
        torch.nn.Module.__init__(self)

    def flow(
            self
    ):
        """
        Return the message-passing flow direction of the first MessagePassing layer found.
        Flow string, e.g. 'source_to_target'; defaults to 'source_to_target' if no MP layer found.
        """
        for module in self.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def get_neurons(
            self
    ) -> list:
        """
        Return the output size of each MessagePassing convolution layer.
        List of neuron counts [n_1, n_2, ..., n_k], one per MP layer with parameters.
        """
        neurons = []

        for module in self.modules():
            if isinstance(module, MessagePassing) and len(module.state_dict()) > 0:
                state_dict = module.state_dict()
                state_dict_reversed_gen = reversed(module.state_dict())
                k = next(state_dict_reversed_gen)
                while not state_dict[k].size():
                    k = next(state_dict_reversed_gen)
                if not state_dict[k].size():
                    n_neurons = neurons[-1]
                else:
                    n_neurons = state_dict[k].shape[0]
                neurons.append(n_neurons)
        return neurons

    def get_weights(
            self,
            tensor_size_limit: Union[int, torch.Tensor] = None
    ) -> dict:
        """
        Return model weights as a nested dict for the frontend.

        Large tensors are replaced with their shape string when they exceed tensor_size_limit.

        Args:
            tensor_size_limit (Union[int, torch.Tensor]): Element count limit; tensors above this
                threshold are represented as shape strings. Default value: `None`.

        Returns:
            Nested dict mirroring the state_dict key hierarchy with tensor values or shape strings.
        """
        try:
            state_dict = self.state_dict()
        except AttributeError:
            state_dict = {}

        model_data = {}
        for key, value in state_dict.items():
            part = model_data
            sub_keys = key.split('.')
            for k in sub_keys[:-1]:
                if k not in part:
                    part[k] = {}
                part = part[k]

            k = sub_keys[-1]
            if type(value) == UninitializedParameter:
                part[k] = '?'
            else:
                size = 1
                for dim in value.shape:
                    size *= dim
                if tensor_size_limit and size > tensor_size_limit:  # Tensor is too big - return just its shape
                    part[k] = 'x'.join(str(d) for d in value.shape)
                else:
                    part[k] = value.numpy().tolist()
        return model_data


class FrameworkGNNConstructor(
    GNNConstructorTorch
):
    """
    A class that uses metaprogramming to form a wide variety of models using the 'structure' variable in json format.
    Inherited from the GNNConstructorTorch class.
    """

    def __init__(
            self,
            model_config: Union[ModelConfig, ConfigPattern] = None,
    ):
        """
        Args:
            model_config (Union[ModelConfig, ConfigPattern]): Model structure configuration.
                Default value: `None`.
        """
        super().__init__()

        self.model_info = None  # Indices of first and last layers at node, graph, and decoder levels
        self.model_config = model_config
        self.structure = self.model_config.structure
        self.n_layers = len(self.structure)
        self.conn_dict = {}  # skip-connections
        self.embedding_levels_by_layers = []
        self.num_hops = None
        self._save_emb_flag = False

        # Hooks can be used for operations graph construction
        self._my_forward_hooks: Dict[int, Callable] = OrderedDict()

        self._parse_structure()
        self._check_model_structure(self.structure)

    def _parse_structure(
            self
    ):
        """
        Parse structure layer by layer and assign attributes to torch.nn.Module
        """
        with open(MODULES_PARAMETERS_PATH) as f:
            self.modules_info = json.load(f)

        for i, elem in enumerate(self.structure):
            self.embedding_levels_by_layers.append(elem['label'])
            layer_name_prefix = ''
            if elem['label'] == 'd':
                layer_name_prefix = 'decoder_'

            # print(elem['layer']['layer_name'])
            if 'function' in elem:
                assert not 'layer' in elem, "Model structure item can be either 'function' or 'layer', not both"
                function_name = elem['function']['function_name']
                function_kwargs = elem['function']['function_kwargs'] or {}
                function_class = import_by_name(
                    self.modules_info[function_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                    self.modules_info[function_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                )
                function_init_class = function_class(**function_kwargs)
                setattr(self, f"{layer_name_prefix}{function_name}_{i}", function_init_class)

            else:  # layer
                layer_name = elem['layer']['layer_name']
                layer_kwargs = elem['layer']['layer_kwargs'] or {}
                if 'GINConv' == layer_name:
                    gin_seq = torch.nn.Sequential()
                    for j, gin_elem in enumerate(elem['layer']['gin_seq']):
                        layer_class = import_by_name(
                            self.modules_info[gin_elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                            self.modules_info[gin_elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                        )
                        layer_init_class = layer_class(**gin_elem['layer']['layer_kwargs'])
                        gin_seq.add_module(f"{gin_elem['layer']['layer_name']}{i}_{j}", layer_init_class)
                        if 'batchNorm' in gin_elem:
                            batch_norm_class = import_by_name(gin_elem['batchNorm']['batchNorm_name'],
                                                              ["torch.nn"])
                            if gin_elem['batchNorm']['batchNorm_kwargs'] is not None:
                                batch_norm = batch_norm_class(
                                    **gin_elem['batchNorm']['batchNorm_kwargs'])
                            else:
                                batch_norm = batch_norm_class()
                            gin_seq.add_module(f'batchNorm{i}_{j}', batch_norm)
                        if 'activation' in gin_elem:
                            activation_class = import_by_name(gin_elem['activation']['activation_name'],
                                                              ["torch.nn"])
                            if gin_elem['activation']['activation_kwargs'] is not None:
                                activation = activation_class(
                                    **gin_elem['activation']['activation_kwargs'])
                            else:
                                activation = activation_class()
                            gin_seq.add_module(f'activation{i}_{j}', activation)
                    gin_class = import_by_name(
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                    )
                    gin = gin_class(nn=gin_seq, **layer_kwargs)
                    setattr(self, f"{layer_name}_{i}", gin)

                # Not GINConv
                elif self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY]["need_full_gnn_flag"]:
                    layer_class = import_by_name(
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                    )
                    custom_layer = layer_class(id(self), f"{layer_name}_{i}", **layer_kwargs)
                    setattr(self, f"{layer_name_prefix}{layer_name}_{i}", custom_layer)

                else:
                    layer_class = import_by_name(
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                        self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                    )
                    layer_init_class = layer_class(**layer_kwargs)
                    setattr(self, f"{layer_name_prefix}{layer_name}_{i}", layer_init_class)

            # FIXME can we have function + batchNorm etc ?
            if 'batchNorm' in elem:
                batch_norm_class = import_by_name(elem['batchNorm']['batchNorm_name'], ["torch.nn"])
                if elem['batchNorm']['batchNorm_kwargs'] is not None:
                    batch_norm = batch_norm_class(**elem['batchNorm']['batchNorm_kwargs'])
                else:
                    batch_norm = batch_norm_class()
                setattr(self, f'{layer_name_prefix}batchNorm_{i}', batch_norm)

            if 'activation' in elem:
                activation_class = import_by_name(elem['activation']['activation_name'],
                                                  ["torch.nn"])
                if elem['activation']['activation_kwargs'] is not None:
                    activation = activation_class(**elem['activation']['activation_kwargs'])
                else:
                    activation = activation_class()
                setattr(self, f'{layer_name_prefix}activation_{i}', activation)

            if 'dropout' in elem:
                dropout_class = import_by_name(elem['dropout']['dropout_name'], ["torch.nn"])
                if elem['dropout']['dropout_kwargs'] is not None:
                    dropout = dropout_class(**elem['dropout']['dropout_kwargs'])
                else:
                    dropout = dropout_class()
                setattr(self, f'{layer_name_prefix}dropout_{i}', dropout)

            if 'connections' in elem:
                for con in elem['connections']:
                    if (i, con['into_layer']) not in self.conn_dict:
                        self.conn_dict[(i, con['into_layer'])] = [
                            copy.deepcopy(con['connection_kwargs'])]
                    else:
                        self.conn_dict[(i, con['into_layer'])].append(
                            copy.deepcopy(con['connection_kwargs']))
        self.model_manager_restrictions = set()

    def _check_model_structure(
            self,
            structure: Union[dict, ModelStructureConfig],
    ):
        """
        Validate the model structure against layer restrictions and populate model_info.

        Args:
            structure (Union[dict, ModelStructureConfig]): Layer structure to validate.
        """
        with open(CUSTOM_LAYERS_INFO_PATH) as f:
            correctness_info = json.load(f)
        allowable_transitions = set(correctness_info["allowable_transitions"])
        for key, elem in self.conn_dict.items():
            if f"{self.embedding_levels_by_layers[key[0]]}{self.embedding_levels_by_layers[key[1]]}" \
                    not in allowable_transitions:
                raise GNNConstructorError(f"Not allowable transitions in connection between layers {key}")
        self.model_info = {
            "first_node_layer_ind": None,
            "last_node_layer_ind": None,
            "first_graph_layer_ind": None,
            "last_graph_layer_ind": None,
            "first_decoder_layer_ind": None,
            "last_decoder_layer_ind": None,
        }
        for i, elem in enumerate(self.embedding_levels_by_layers):
            if i != len(
                    self.embedding_levels_by_layers) - 1 and \
                    f"{self.embedding_levels_by_layers[i]}{self.embedding_levels_by_layers[i + 1]}" \
                    not in allowable_transitions:
                raise GNNConstructorError(f"Not allowable transitions between layers ({i}, {i + 1})")
            if elem == 'n' and self.model_info["first_node_layer_ind"] is None:
                self.model_info["first_node_layer_ind"] = i
                if i == len(self.embedding_levels_by_layers) - 1:
                    self.model_info["last_node_layer_ind"] = i
            elif elem == 'n' and i == len(self.embedding_levels_by_layers) - 1:
                self.model_info["last_node_layer_ind"] = i
            elif elem == 'g' and self.model_info["first_graph_layer_ind"] is None:
                self.model_info["first_graph_layer_ind"] = i
                self.model_info["last_node_layer_ind"] = i - 1
                if i == len(self.embedding_levels_by_layers) - 1:
                    self.model_info["last_graph_layer_ind"] = i
            elif elem == 'g' and i == len(self.embedding_levels_by_layers) - 1:
                self.model_info["last_graph_layer_ind"] = i
            elif elem == 'd' and self.model_info["first_decoder_layer_ind"] is None:
                self.model_info["first_decoder_layer_ind"] = i
                self.model_info["last_node_layer_ind"] = i - 1
            if elem == 'd' and i == len(self.embedding_levels_by_layers) - 1:
                self.model_info["last_decoder_layer_ind"] = i

        _model_strong_restrictions = set()
        _model_manager_restrictions = set()
        for i, elem in enumerate(structure):
            if 'function' in elem:
                layer_name = elem['function']['function_name']
            else:
                layer_name = elem['layer']['layer_name']

            # Check if layer is known (listed in restrictions file)
            if layer_name not in correctness_info["layers_restrictions"].keys():
                raise GNNConstructorError(
                    f"An invalid layer {layer_name} is used in the model structure:\n{elem}")

            layers_restrictions = correctness_info["layers_restrictions"][layer_name]

            # Check if layer requires a specified model manager
            if len(layers_restrictions["model_manager_restrictions"]) > 0:
                if len(_model_manager_restrictions) > 0:
                    _model_manager_restrictions = _model_manager_restrictions.intersection(
                        layers_restrictions["model_manager_restrictions"])
                    if len(_model_manager_restrictions) == 0:
                        raise GNNConstructorError(
                            f"Model structure cannot use layer {layer_name}, because there is no "
                            f"suitable model manager for such a model. Write a new model manager "
                            f"and/or add an appropriate model manager to the {layer_name} "
                            f"layer restrictions")
                else:
                    _model_manager_restrictions = set(layers_restrictions["model_manager_restrictions"])

            # Check if model structure restrictions do not contradict
            if not _model_strong_restrictions.isdisjoint(layers_restrictions["strong_restrictions"]):
                # FIXME Kirill, _model_strong_restrictions is always empty - how can we get here?
                raise GNNConstructorError(
                    f"Model structure cannot use layers with the same strong constraints. "
                    f"{_model_strong_restrictions.intersection(layers_restrictions['strong_restrictions'])}")

            # Check that layer label is allowed
            if elem['label'] not in layers_restrictions["valid_label"]:
                raise GNNConstructorError(f"Invalid label {elem['label']} for layer {layer_name}")

            # Check that restrictions on first and last node/graph/decoder layers are met
            layer_strong_restrictions = layers_restrictions["strong_restrictions"]
            if len(layer_strong_restrictions) > 0:
                if 'first_model' in layer_strong_restrictions and i != 0:
                    raise GNNConstructorError(f"Layer {layer_name} must be the first in the model")
                if 'last_model' in layer_strong_restrictions and i != len(self.embedding_levels_by_layers) - 1:
                    raise GNNConstructorError(f"Layer {layer_name} must be the last in the model")

                if 'first_node' in layer_strong_restrictions and i != self.model_info["first_node_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the first in the model node level")
                if 'last_node' in layer_strong_restrictions and i != self.model_info["last_node_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the last in the model node level")

                if 'first_graph' in layer_strong_restrictions and i != self.model_info["first_graph_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the first in the model graph level")
                if 'last_graph' in layer_strong_restrictions and i != self.model_info["last_graph_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the last in the model graph level")

                if 'first_decoder' in layer_strong_restrictions and i != self.model_info["first_decoder_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the first in the model decoder level")
                if 'last_decoder' in layer_strong_restrictions and i != self.model_info["last_decoder_layer_ind"]:
                    raise GNNConstructorError(f"Layer {layer_name} must be the last in the model decoder level")

                _model_strong_restrictions.update(layer_strong_restrictions)
        self.model_manager_restrictions = _model_manager_restrictions

    def get_all_layer_embeddings(
            self,
            *args,
            **kwargs
    ) -> dict:
        """
        Run the forward pass in embedding-capture mode and return per-layer output tensors.

        Returns:
            Dict mapping layer index to the output tensor of that layer.
        """
        self._save_emb_flag = True
        layer_emb_dict = self(*args, **kwargs)
        self._save_emb_flag = False
        return layer_emb_dict

    def register_my_forward_hook(
            self, hook: Callable[..., None]
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._my_forward_hooks[handle.id] = hook
        return handle

    def forward(
            self,
            *args,
            **kwargs
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Execute the node/graph-level forward pass, applying skip-connections where configured.

        Returns:
            Output tensor, or a dict of layer-index → tensor when in embedding-capture mode.
        """
        layer_ind = -1
        tensor_storage = {}
        dim_cat = 0
        layer_emb_dict = {}
        save_emb_flag = self._save_emb_flag

        x, edge_index, batch, edge_weight = self.arguments_read(*args, **kwargs)
        device = next(self.parameters()).device
        x, edge_index, batch, edge_weight = self.move_to_device(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_weight=edge_weight,
            device=device
        )
        feat = x
        # print(list(self.__dict__['_modules'].items()))
        for elem in list(self.__dict__['_modules'].items()):
            if elem[0].startswith('decoder_'):  # Omit decoder layers
                continue
            layer_name, curr_layer_ind = elem[0].split('_')
            curr_layer_ind = int(curr_layer_ind)
            inp = x
            loc_flag = False
            if curr_layer_ind != layer_ind:
                if save_emb_flag:
                    loc_flag = True
                zeroing_x_flag = False
                for key, value in self.conn_dict.items():
                    if key[0] == layer_ind and layer_ind not in tensor_storage:
                        tensor_storage[layer_ind] = torch.clone(x)
                layer_ind = curr_layer_ind
                x_copy = torch.clone(x)
                connection_tensor = torch.empty(0, device=x_copy.device)
                x_dict = {}
                for key, value in self.conn_dict.items():
                    if key[1] == curr_layer_ind:
                        if key[1] - key[0] == 1:
                            zeroing_x_flag = True
                        for con in value:
                            aggregation_type = con.get('aggregation_type', 'cat')

                            if aggregation_type == 'cat':
                                if connection_tensor is None:
                                    connection_tensor = tensor_storage[key[0]]
                                else:
                                    if self.embedding_levels_by_layers[key[1]] == 'n' and \
                                            self.embedding_levels_by_layers[key[0]] == 'n':
                                        connection_tensor = torch.cat((connection_tensor, tensor_storage[key[0]]), 1)
                                        dim_cat = 1
                                    elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                            self.embedding_levels_by_layers[key[0]] == 'g':
                                        connection_tensor = torch.cat((connection_tensor, tensor_storage[key[0]]), 0)
                                        dim_cat = 0
                                    elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                            self.embedding_levels_by_layers[key[0]] == 'n':
                                        con_pool = import_by_name(con['pool']['pool_type'], ["torch_geometric.nn"])
                                        tensor_after_pool = con_pool(tensor_storage[key[0]], batch)
                                        connection_tensor = torch.cat((connection_tensor, tensor_after_pool), 1)
                                        dim_cat = 1
                                    else:
                                        raise GNNConstructorError(
                                            f"Connection from layer type {self.embedding_levels_by_layers[curr_layer_ind - 1]} to "
                                            f"layer type {self.embedding_levels_by_layers[curr_layer_ind]} is not supported now"
                                        )

                            elif aggregation_type == 'stack':
                                if self.embedding_levels_by_layers[key[1]] == 'n' and self.embedding_levels_by_layers[
                                                                                                        key[0]] == 'n':
                                    x_dict[f'skip_{key[0]}'] = tensor_storage[key[0]]
                                elif self.embedding_levels_by_layers[key[1]] == 'g' and self.embedding_levels_by_layers[
                                                                                                        key[0]] == 'g':
                                    x_dict[f'skip_{key[0]}'] = tensor_storage[key[0]]
                                elif self.embedding_levels_by_layers[key[1]] == 'g' and self.embedding_levels_by_layers[
                                                                                                        key[0]] == 'n':
                                    con_pool = import_by_name(con['pool']['pool_type'], ["torch_geometric.nn"])
                                    tensor_after_pool = con_pool(tensor_storage[key[0]], batch)
                                    x_dict[f'skip_{key[0]}'] = tensor_after_pool
                                else:
                                    raise GNNConstructorError(
                                        f"Connection from layer type {self.embedding_levels_by_layers[curr_layer_ind - 1]} to "
                                        f"layer type {self.embedding_levels_by_layers[curr_layer_ind]} is not supported now"
                                    )
                            else:
                                raise ValueError(f"Unknown aggregation type: {aggregation_type}")
                if len(x_dict) > 0:  # stack
                    x_dict[f'prev_{curr_layer_ind - 1}'] = x_copy
                    x = x_dict
                else:  # cat
                    if zeroing_x_flag:
                        x = connection_tensor
                    else:
                        x = torch.cat((x_copy, connection_tensor), dim_cat)

            # QUE Kirill, maybe we should not off UserWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # mid = x
                if layer_name in self.modules_info:
                    code_str = f"getattr(self, elem[0])({self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY]['forward_parameters']})"
                    x = eval(f"{code_str}")
                else:
                    x = getattr(self, elem[0])(x)
            if loc_flag:
                layer_emb_dict[layer_ind] = torch.clone(x)

            # out = x
            # if self._my_forward_hooks:
            #     for hook in self._my_forward_hooks.values():
            #         hook(self, curr_layer_ind, feat, edge_index, inp, mid, out)
        if save_emb_flag:
            # layer_emb_dict[layer_ind] = torch.clone(x)
            return layer_emb_dict
        return x

    def decode(
            self,
            src: torch.Tensor,
            dst: torch.Tensor,
            # batch
    ) -> torch.Tensor:
        """
        Execute the decoder-level forward pass over edge node-embedding pairs.

        Args:
            src (torch.Tensor): Source node embeddings.
            dst (torch.Tensor): Destination node embeddings.

        Returns:
            Edge score tensor (squeezed to remove trailing size-1 dimensions).
        """
        layer_ind = -1
        # tensor_storage = {}
        # dim_cat = 0
        layer_emb_dict = {}
        save_emb_flag = self._save_emb_flag

        # TODO what about device

        is_first_layer = True
        for elem in list(self.__dict__['_modules'].items()):
            if not elem[0].startswith('decoder_'):  # Omit non-decoder layers
                continue
            _, layer_name, curr_layer_ind = elem[0].split('_')
            curr_layer_ind = int(curr_layer_ind)
            loc_flag = False
            if curr_layer_ind != layer_ind:
                if save_emb_flag:
                    loc_flag = True
                zeroing_x_flag = False
                # for key, value in self.conn_dict.items():
                #     if key[0] == layer_ind and layer_ind not in tensor_storage:
                #         tensor_storage[layer_ind] = torch.clone(x)
                layer_ind = curr_layer_ind

            # QUE Kirill, maybe we should not off UserWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # mid = x
                if layer_name in self.modules_info:
                    code_str = f"getattr(self, elem[0])({self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY]['forward_parameters']})"
                    x = eval(f"{code_str}")
                else:
                    if is_first_layer:
                        x = getattr(self, elem[0])(src, dst)
                    else:
                        x = getattr(self, elem[0])(x)
            if loc_flag:
                layer_emb_dict[layer_ind] = torch.clone(x)

            is_first_layer = False

        # Remove dimension of size=1 after last layer to get edge score as a single number not list
        x = torch.squeeze(x)

        if save_emb_flag:
            layer_emb_dict[layer_ind] = torch.clone(x)
            return layer_emb_dict

        return x


    def reset_parameters(
            self
    ) -> None:
        """ Reset parameters of all sub-modules that expose a reset_parameters method.
        """
        for elem in list(self.__dict__['_modules'].items()):
            if {'reset_parameters'}.issubset(dir(getattr(self, elem[0]))):
                getattr(self, elem[0]).reset_parameters()

    def get_architecture(
            self
    ) -> Union[dict, ModelStructureConfig]:
        return self.structure

    def get_num_hops(
            self
    ) -> int:
        """
        Return the total number of graph convolution hops (APPNP contributes K hops).

        Returns:
            Number of message-passing hops.
        """
        if self.num_hops is None:
            num_hops = 0
            for module in self.modules():
                if isinstance(module, MessagePassing):
                    if isinstance(module, import_by_name('APPNP', ['torch_geometric.nn'])):
                        num_hops += module.K
                    else:
                        num_hops += 1
            self.num_hops = num_hops
            return num_hops
        else:
            return self.num_hops

    def get_predictions(
            self,
            *args,
            edge_out: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Return class probability distribution (node/graph) or sigmoid score (edge prediction).

        Args:
            *args: Forwarded to the model's forward pass.
            edge_out (torch.Tensor): Pre-computed edge logits for edge prediction task.
                If provided, applies sigmoid instead of softmax. Default value: `None`.
            **kwargs: Forwarded to the model's forward pass.

        Returns:
            Probability tensor.
        """
        # FIXME Kirill. tmp fix of AttributeError: 'dict' object has no attribute 'softmax' for SubgraphX
        # self._save_emb_flag = False
        if edge_out is not None:
            # Edge prediction task. Apply threshold
            return edge_out.sigmoid()
        else:
            return self(*args, **kwargs).softmax(dim=-1)
        # return self.forward(*args, **kwargs)

    def get_parameters(
            self
    ) -> Iterator:
        return self.parameters()

    def get_answer(
            self,
            *args,
            threshold: float = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Return hard predictions (class indices or binary edge decisions).

        Args:
            *args: Forwarded to get_predictions.
            threshold (float): For edge prediction: predict True when sigmoid score exceeds threshold.
                If None, returns argmax over class probabilities. Default value: `None`.
            **kwargs: Forwarded to get_predictions.

        Returns:
            Tensor of predicted class indices or boolean edge decisions.
        """
        if threshold is not None:
            # Edge prediction task. Apply threshold
            return self.get_predictions(*args, **kwargs) > threshold
        else:
            return self.get_predictions(*args, **kwargs).argmax(dim=1)

    def suitable_model_managers(
            self
    ) -> set:
        return self.model_manager_restrictions

    @staticmethod
    def move_to_device(
            x: torch.Tensor = None,
            edge_index: torch.Tensor = None,
            batch: Type = None,
            edge_weight: torch.Tensor = None,
            device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Type, torch.Tensor]:
        """
        Move all graph tensors to the specified device.

        Args:
            x (torch.Tensor): Node feature matrix. Default value: `None`.
            edge_index (torch.Tensor): Edge index tensor. Default value: `None`.
            batch (Type): Batch assignment vector. Default value: `None`.
            edge_weight (torch.Tensor): Edge weight tensor. Default value: `None`.
            device (torch.device): Target device. Defaults to CUDA if available. Default value: `None`.

        Returns:
            Tuple of (x, edge_index, batch, edge_weight) on the target device.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def to_device(tensor):
            return tensor.to(device) if tensor is not None else None

        x = to_device(x)
        edge_index = to_device(edge_index)
        batch = to_device(batch)
        edge_weight = to_device(edge_weight)

        return x, edge_index, batch, edge_weight

    @staticmethod
    def arguments_read(
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Type, torch.Tensor]:
        """
        Extract (x, edge_index, batch, edge_weight) from any supported calling convention.
        Accepts a PyG Data object, keyword arguments, or positional arguments (2, 3, or 4 tensors).

        !! ATTENTION: Must not be changed !!

        Returns:
            Tuple of (x, edge_index, batch, edge_weight) as torch Tensors.
        """

        data = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                edge_weight = kwargs.get('edge_weight', None)
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            else:
                if len(args) == 1:
                    args = args[0]
                    if 'x' in args and 'edge_index' in args:
                        x, edge_index = args.x, args.edge_index
                    else:
                        raise ValueError(f"forward's args should contain x and 3"
                                         f" edge_index Tensors but {args.keys} doesn't content this Tensors")
                    if 'batch' in args:
                        batch = args.batch
                    else:
                        batch = torch.zeros(args.x.shape[0], dtype=torch.int64, device=x.device)
                    if 'edge_weight' in args:
                        edge_weight = args.edge_weight
                    else:
                        edge_weight = None
                else:
                    if len(args) == 2:
                        x, edge_index = args[0], args[1]
                        batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
                        edge_weight = None
                    elif len(args) == 3:
                        x, edge_index, batch = args[0], args[1], args[2]
                        edge_weight = None
                    elif len(args) == 4:
                        x, edge_index, batch, edge_weight = args[0], args[1], args[2], args[3]
                    else:
                        raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")

        else:
            if hasattr(data, "edge_weight"):
                x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
            else:
                x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, None

        return x, edge_index, batch, edge_weight
