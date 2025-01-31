import copy
import json
import warnings
from collections import OrderedDict
from typing import Dict, Callable, Union, Iterator, Type
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils import hooks
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing

from aux.utils import import_by_name, CUSTOM_LAYERS_INFO_PATH, MODULES_PARAMETERS_PATH, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY
from aux.configs import ModelConfig, CONFIG_CLASS_NAME, ModelStructureConfig


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
        :param model_config:
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

    def get_all_layer_embeddings(
            self
    ):
        """
        :return: vectors representing the input data from the outputs of each layer of the neural network
        """
        raise NotImplementedError("get_all_layer_embeddings can't be called, because it is not implemented")

    def get_architecture(
            self
    ):
        """
        :return: the architecture of the model, for display on the front
        """
        raise NotImplementedError("get_architecture can't be called, because it is not implemented")

    def get_num_hops(
            self
    ):
        """
        :return: the number of graph convolution layers. Required for some model interpretation algorithms to work
        """
        raise NotImplementedError("get_num_hops can't be called, because it is not implemented")

    def reset_parameters(
            self
    ):
        """
        resets all model parameters. Required for the reset button on the front to work.
        """
        raise NotImplementedError("reset_parameters can't be called, because it is not implemented")

    def get_predictions(
            self
    ):
        """
        :return: a vector of estimates for the distribution of input data by class.
        Required for some interpretation algorithms to work.
        Does not require a mandatory redefinition of def forward.
        """
        raise NotImplementedError("get_predictions can't be called, because it is not implemented")

    def get_parameters(
            self
    ):
        """
        :return: matrix with model parameters
        """
        raise NotImplementedError("get_predictions can't be called, because it is not implemented")

    def get_answer(
            self
    ):
        """
        :return: an answer to which class the input belongs to.
        Required for some interpretation methods to work.
        Does not require redefinition of def forward or get_predictions.
        """
        raise NotImplementedError("get_answer can't be called, because it is not implemented")

    def get_name(
            self,
            obj_name_flag: bool = False,
            **kwargs
    ):
        gnn_name = self.model_config.to_saveable_dict().copy()
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
        :return: a set of names of suitable model manager classes.
        Model manager classes must be inherited from the GNNModelManager class
        """
        raise NotImplementedError("suitable_model_managers can't be called, because it is not implemented")

    # === Permanent methods (not to be overwritten)

    def get_hash(
            self
    ) -> str:
        gnn_name = self.get_name()
        json_object = json.dumps(gnn_name)
        gnn_name_hash = hash_data_sha256(json_object.encode('utf-8'))
        return gnn_name_hash

    def get_full_info(
            self,
            tensor_size_limit: int=None
    ) -> dict:
        """ Get available info about model for frontend
        """
        # FIXMe architecture and weights can be not accessible
        result = {}
        try:
            result["architecture"] = self.get_architecture()
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
        """ Flow direction of message passing, usually 'source_to_target'
        """
        for module in self.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def get_neurons(
            self
    ):
        """ Return number of neurons of each convolution layer as list: [n_1, n_2, ..., n_k]
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
            tensor_size_limit: str = None
    ):
        """
        Get model weights calling torch.nn.Module.state_dict() to draw them on the frontend.
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
            model_config: ModelConfig = None,
    ):
        """
        :param model_config: description of the gnn structure
        """
        super().__init__()

        self.model_info = None
        self.model_config = model_config
        self.structure = self.model_config.structure
        self.n_layers = len(self.structure)
        self.conn_dict = {}
        self.embedding_levels_by_layers = []
        self.num_hops = None
        self._save_emb_flag = False

        # Hooks can be used for operations graph construction
        self._my_forward_hooks: Dict[int, Callable] = OrderedDict()

        with open(MODULES_PARAMETERS_PATH) as f:
            self.modules_info = json.load(f)

        for i, elem in enumerate(self.structure):
            self.embedding_levels_by_layers.append(elem['label'])
            # print(elem['layer']['layer_name'])

            if 'GINConv' == elem['layer']['layer_name']:
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
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                )
                if elem['layer']['layer_kwargs'] is not None:
                    gin = gin_class(nn=gin_seq, **elem['layer']['layer_kwargs'])
                else:
                    gin = gin_class(nn=gin_seq)
                setattr(self, f"{elem['layer']['layer_name']}_{i}", gin)
            elif self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY]["need_full_gnn_flag"]:
                layer_class = import_by_name(
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                )
                custom_layer = layer_class(id(self), f"{elem['layer']['layer_name']}_{i}",
                                           **elem['layer']['layer_kwargs'])
                setattr(self, f"{elem['layer']['layer_name']}_{i}", custom_layer)
            else:
                layer_class = import_by_name(
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][0],
                    self.modules_info[elem['layer']['layer_name']][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY][1]
                )
                layer_init_class = layer_class(**elem['layer']['layer_kwargs'])
                setattr(self, f"{elem['layer']['layer_name']}_{i}", layer_init_class)

            if 'batchNorm' in elem:
                batch_norm_class = import_by_name(elem['batchNorm']['batchNorm_name'], ["torch.nn"])
                if elem['batchNorm']['batchNorm_kwargs'] is not None:
                    batch_norm = batch_norm_class(**elem['batchNorm']['batchNorm_kwargs'])
                else:
                    batch_norm = batch_norm_class()
                setattr(self, 'batchNorm_%s' % i, batch_norm)

            if 'activation' in elem:
                activation_class = import_by_name(elem['activation']['activation_name'],
                                                  ["torch.nn"])
                if elem['activation']['activation_kwargs'] is not None:
                    activation = activation_class(**elem['activation']['activation_kwargs'])
                else:
                    activation = activation_class()
                setattr(self, 'activation_%s' % i, activation)

            if 'dropout' in elem:
                dropout_class = import_by_name(elem['dropout']['dropout_name'], ["torch.nn"])
                if elem['dropout']['dropout_kwargs'] is not None:
                    dropout = dropout_class(**elem['dropout']['dropout_kwargs'])
                else:
                    dropout = dropout_class()
                setattr(self, 'dropout_%s' % i, dropout)

            if 'connections' in elem:
                for con in elem['connections']:
                    if (i, con['into_layer']) not in self.conn_dict:
                        self.conn_dict[(i, con['into_layer'])] = [
                            copy.deepcopy(con['connection_kwargs'])]
                    else:
                        self.conn_dict[(i, con['into_layer'])].append(
                            copy.deepcopy(con['connection_kwargs']))
        self.model_manager_restrictions = set()
        self._check_model_structure(self.structure)

    def _check_model_structure(
            self,
            structure: Union[dict, ModelStructureConfig],
    ):
        with open(CUSTOM_LAYERS_INFO_PATH) as f:
            information_check_correctness_models = json.load(f)
        allowable_transitions = set(information_check_correctness_models["allowable_transitions"])
        for key, elem in self.conn_dict.items():
            if f"{self.embedding_levels_by_layers[key[0]]}{self.embedding_levels_by_layers[key[1]]}" \
                    not in allowable_transitions:
                raise Exception(f"Not allowable transitions in connection between layers {key}")
        self.model_info = {
            "first_node_layer_ind": None,
            "last_node_layer_ind": None,
            "first_graph_layer_ind": None,
            "last_graph_layer_ind": None,
        }
        for i, elem in enumerate(self.embedding_levels_by_layers):
            if i != len(
                    self.embedding_levels_by_layers) - 1 and \
                    f"{self.embedding_levels_by_layers[i]}{self.embedding_levels_by_layers[i + 1]}" \
                    not in allowable_transitions:
                raise Exception(f"Not allowable transitions between layers ({i}, {i + 1})")
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
        model_strong_restrictions = set()
        model_manager_restrictions = set()
        for i, elem in enumerate(structure):
            layer_name = elem['layer']['layer_name']
            if layer_name not in information_check_correctness_models["layers_restrictions"].keys():
                raise Exception(f"An invalid layer {layer_name} is used in the model structure.\n{elem}")
            else:
                if len(information_check_correctness_models["layers_restrictions"][layer_name]
                       ["model_manager_restrictions"]) > 0:
                    if len(model_manager_restrictions) > 0:
                        model_manager_restrictions = model_manager_restrictions.intersection(
                            information_check_correctness_models["layers_restrictions"][layer_name]
                            ["model_manager_restrictions"])
                        if len(model_manager_restrictions) == 0:
                            raise Exception(f"Model structure cannot use layer {layer_name}, because there is no "
                                            f"suitable model manager for such a model. Write a new model manager "
                                            f"and/or add an appropriate model manager to the {layer_name} "
                                            f"layer restrictions")
                    else:
                        model_manager_restrictions = set(information_check_correctness_models["layers_restrictions"]
                                                         [layer_name]["model_manager_restrictions"])
                if not model_strong_restrictions.isdisjoint(
                        information_check_correctness_models["layers_restrictions"][layer_name]["strong_restrictions"]):
                    raise Exception(f"Model structure cannot use layers with the same strong constraints. "
                                    f"{model_strong_restrictions.intersection(information_check_correctness_models['layers_restrictions'][layer_name]['strong_restrictions'])}")
                else:
                    layer_strong_restrictions = information_check_correctness_models["layers_restrictions"][layer_name][
                        "strong_restrictions"]
                    layer_valid_label = information_check_correctness_models["layers_restrictions"][layer_name][
                        "valid_label"]
                    if elem['label'] not in layer_valid_label:
                        raise Exception(f"Invalid label {elem['label']} for layer {layer_name}")
                    if len(layer_strong_restrictions) > 0:
                        if 'first_model' in layer_strong_restrictions and i != 0:
                            raise Exception(f"Layer {layer_name} must be the first in the model")
                        if 'last_model' in layer_strong_restrictions and i != len(self.embedding_levels_by_layers) - 1:
                            raise Exception(f"Layer {layer_name} must be the last in the model")
                        if 'first_node' in layer_strong_restrictions and i != self.model_info["first_node_layer_ind"]:
                            raise Exception(f"Layer {layer_name} must be the first in the model node level")
                        if 'last_node' in layer_strong_restrictions and i != self.model_info["last_node_layer_ind"]:
                            raise Exception(f"Layer {layer_name} must be the last in the model node level")
                        if 'first_graph' in layer_strong_restrictions and i != self.model_info["first_graph_layer_ind"]:
                            raise Exception(f"Layer {layer_name} must be the first in the model graph level")
                        if 'last_graph' in layer_strong_restrictions and i != self.model_info["last_graph_layer_ind"]:
                            raise Exception(f"Layer {layer_name} must be the last in the model graph level")
                        model_strong_restrictions.update(layer_strong_restrictions)
        self.model_manager_restrictions = model_manager_restrictions

    def get_all_layer_embeddings(
            self,
            *args,
            **kwargs
    ) -> dict:
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
    ) -> torch.Tensor:
        layer_ind = -1
        tensor_storage = {}
        dim_cat = 0
        layer_emb_dict = {}
        save_emb_flag = self._save_emb_flag

        x, edge_index, batch, edge_weight = self.arguments_read(*args, **kwargs)
        feat = x
        # print(list(self.__dict__['_modules'].items()))
        for elem in list(self.__dict__['_modules'].items()):
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
                for key, value in self.conn_dict.items():
                    if key[1] == curr_layer_ind:
                        if key[1] - key[0] == 1:
                            zeroing_x_flag = True
                        for con in self.conn_dict[key]:
                            if self.embedding_levels_by_layers[key[1]] == 'n' and \
                                    self.embedding_levels_by_layers[key[0]] == 'n':
                                connection_tensor = torch.cat((connection_tensor,
                                                               tensor_storage[key[0]]), 1)
                                dim_cat = 1
                            elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                    self.embedding_levels_by_layers[key[0]] == 'g':
                                connection_tensor = torch.cat((connection_tensor,
                                                               tensor_storage[key[0]]), 0)
                                dim_cat = 0
                            elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                    self.embedding_levels_by_layers[key[0]] == 'n':
                                con_pool = import_by_name(con['pool']['pool_type'],
                                                          ["torch_geometric.nn"])
                                tensor_after_pool = con_pool(tensor_storage[key[0]], batch)
                                connection_tensor = torch.cat((connection_tensor,
                                                               tensor_after_pool), 1)
                                dim_cat = 1
                            else:
                                raise Exception(
                                    "Connection from layer type "
                                    f"{self.embedding_levels_by_layers[curr_layer_ind - 1]} to"
                                    f" layer type {self.embedding_levels_by_layers[curr_layer_ind]}"
                                    "is not supported now")
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

    def reset_parameters(
            self
    ) -> None:
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
            **kwargs
    ) -> torch.Tensor:
        return self(*args, **kwargs).softmax(dim=-1)
        # return self.forward(*args, **kwargs)

    def get_parameters(
            self
    ) -> Iterator:
        return self.parameters()

    def get_answer(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        return self.get_predictions(*args, **kwargs).argmax(dim=1)

    def suitable_model_managers(
            self
    ) -> set:
        return self.model_manager_restrictions

    @staticmethod
    def arguments_read(
            *args, **kwargs
    ) -> [torch.Tensor, torch.Tensor, Type, torch.Tensor]:
        """
        The method is launched when the forward is executed extracts from the variable data or kwargs
        the data necessary to pass the forward: x, edge_index, batch

        !! ATTENTION: Must not be changed !!

        :param args:
        :param kwargs:
        :return: x, edge_index, batch: TORCH TENSORS
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
