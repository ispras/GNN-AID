from typing import Type, Union
import torch
import copy

from gnn_aid.attacks.attack_base import Attacker
from gnn_aid.aux.utils import move_to_same_device
from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.models_builder.model_managers import GNNModelManager

# Nettack imports
from .evasion_attacks_collection.nettack.utils import NettackSurrogate, NettackAttack

# PGD imports
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from gnn_aid.models_builder.models_utils import EdgeMaskingWrapper
from .evasion_attacks_collection.pgd.utils import random_sampling
from gnn_aid.data_structures.graph_modification_artifacts import GraphModificationArtifact, GlobalNodeIndexer
from tqdm import tqdm

# FGSM imports
from gnn_aid.models_builder.models_utils import apply_decorator_to_graph_layers

# ReWatt imports
from .evasion_attacks_collection.rewatt.utils import GraphEnvironment, ReWattPolicyNet, \
    GraphState, ReWattAgent
from ..data_structures import Task


class EvasionAttacker(
    Attacker
):
    """ Base class for all poison attack methods.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()


class EmptyEvasionAttacker(
    EvasionAttacker
):
    """ Just a stub for evasion attack.
    """
    name = "EmptyEvasionAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass


class FGSMAttacker(
    EvasionAttacker
):
    name = "FGSM"

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        """ Availability check for the given dataset and model manager. """
        return True

    def __init__(
            self,
            is_feature_attack: bool = False,
            element_idx: int = 0,
            epsilon: float = 0.5,
    ):
        super().__init__()
        self.is_feature_attack = is_feature_attack
        self.element_idx = element_idx
        self.epsilon = epsilon
        self.grad_aggr_type = 'mean'
        self.attack_diff = GraphModificationArtifact()
        # self.attack_res_misha = None

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor,
            task_type: str = None,
    ):
        task = gen_dataset.dataset_var_config.task
        device = gen_dataset.data.x.device

        if self.is_feature_attack:
            model = model_manager.gnn
            model.eval()

            if task == Task.EDGE_PREDICTION:
                data = gen_dataset.data
                x = data.x
                x.requires_grad = True

                # TODO now support only one edge, mask_tensor?
                edge_label_index = torch.tensor(self.element_idx).unsqueeze(dim=1).to(device)
                edge_label = ((data.edge_index == edge_label_index).all(dim=0).any()).float().unsqueeze(dim=0).to(device)

                node_out = model(data.x, data.edge_index)
                src = node_out[edge_label_index[0]]
                dst = node_out[edge_label_index[1]]
                edge_out = model.decode(src, dst).unsqueeze(dim=0).to(device)

                # TODO use model_manager.loss_function when BCE support
                # loss = model_manager.loss_function(edge_out, edge_label)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(edge_out, edge_label)
                model.zero_grad()
                loss.backward()
                sign_data_grad = x.grad.sign()
                perturbed_data_x = x + self.epsilon * sign_data_grad
                perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
                gen_dataset.data.x = perturbed_data_x.detach()

            elif task == Task.GRAPH_CLASSIFICATION:
                graph_idx = self.element_idx
                x = gen_dataset.dataset[graph_idx].x
                x.requires_grad = True
                edge_index = gen_dataset.dataset[graph_idx].edge_index
                y = gen_dataset.dataset[graph_idx].y

                output = model(x, edge_index)
                loss = model_manager.loss_function(*move_to_same_device(output, y))
                model.zero_grad()
                loss.backward()
                sign_data_grad = x.grad.sign()
                perturbed_data_x = x + self.epsilon * sign_data_grad
                perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
                gen_dataset.dataset[graph_idx].x = perturbed_data_x.detach()

            elif task == Task.NODE_CLASSIFICATION:
                node_idx = self.element_idx
                x = gen_dataset.data.x
                x.requires_grad = True
                edge_index = gen_dataset.data.edge_index
                y = gen_dataset.data.y[node_idx]

                output = model(x, edge_index)[node_idx]
                loss = model_manager.loss_function(*move_to_same_device(output, y))
                model.zero_grad()
                loss.backward()
                sign_data_grad = x.grad.sign()
                perturbed_data_x = x + self.epsilon * sign_data_grad
                perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
                gen_dataset.data.x = perturbed_data_x.detach()

            else:
                raise NotImplementedError

            # if task_type:
            #     gni = GlobalNodeIndexer(gen_dataset.dataset)
            #     for node_idx in tqdm(range(perturbed_data_x.size(0))):
            #         for feature_idx in range(perturbed_data_x.size(1)):
            #             self.attack_diff.change_node_feature(node_idx, feature_idx, perturbed_data_x[gni.to_global(graph_idx, node_idx)][feature_idx].detach().cpu().item())
            # else:
            #     for node_idx in tqdm(range(gen_dataset.data.x.size(0))):
            #         for feature_idx in range(gen_dataset.data.x.size(1)):
            #             self.attack_diff.change_node_feature(node_idx, feature_idx, perturbed_data_x[node_idx][feature_idx].detach().cpu().item())
        else:
            if task.is_edge_level():

                edge_index = gen_dataset.data.edge_index
                y = gen_dataset.data.y
                x = gen_dataset.data.x

                edge_label_index = torch.tensor(self.element_idx).unsqueeze(dim=1).to(device)
                edge_label = ((edge_index == edge_label_index).all(dim=0).any()).float().unsqueeze(dim=0).to(device)
                node_idx_1 = edge_label_index[0].item()
                node_idx_2 = edge_label_index[1].item()

                model = model_manager.gnn
                model.eval()
                num_hops = model.n_layers

                subset, edge_index_subset, inv, edge_mask = k_hop_subgraph(node_idx=[node_idx_1, node_idx_2],
                                                                           num_hops=num_hops,
                                                                           edge_index=edge_index,
                                                                           relabel_nodes=True,
                                                                           directed=False)

                node_idx_1_remap = torch.where(subset == node_idx_1)[0]
                node_idx_2_remap = torch.where(subset == node_idx_2)[0]
                x = x.clone()
                x = x[subset]

                first_layer_name, layer = next(model.named_children())
                apply_decorator_to_graph_layers(model)

                # budget = int(self.epsilon * edge_index_subset.size(1))
                budget = 10

                perturbed_edges = edge_index_subset.clone()
                self_loops_part_size = subset.size(0)

                for _ in tqdm(range(budget)):
                    node_out = model(x, perturbed_edges)
                    src = node_out[node_idx_1_remap]
                    dst = node_out[node_idx_2_remap]
                    edge_out = model.decode(src, dst).unsqueeze(dim=0).to(device)
                    # TODO use model_manager.loss_function when BCE support
                    # loss = -model_manager.loss_function(edge_out, edge_label)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    loss = -criterion(edge_out, edge_label)
                    loss.backward()

                    grad = layer.message_gradients[first_layer_name]
                    if self.grad_aggr_type == 'mean':
                        grad = torch.mean(grad, dim=1)
                    else:
                        raise ValueError(f"Unsupported grad_aggr_type: {self.grad_aggr_type}")

                    if grad.size(0) != perturbed_edges.size(1):  # model use add_self_loops
                        grad = grad[:-self_loops_part_size]

                    max_index = torch.argmax(grad)
                    perturbed_edges = torch.cat((perturbed_edges[:, :max_index], perturbed_edges[:, max_index + 1:]),
                                                dim=1)

                # TODO check correctness
                # Update dataset
                edges_to_keep = edge_index[:, ~edge_mask]
                updated_edge_index = torch.cat([edges_to_keep, perturbed_edges], dim=1)
                gen_dataset.data.edge_index = updated_edge_index
                # self.attack_diff = gen_dataset
                set_a = set(map(tuple, edge_index.T.tolist()))
                set_b = set(map(tuple, updated_edge_index.T.tolist()))
                diff_a = list(set_a - set_b)
                self.attack_diff.remove_edges(diff_a)

            elif task.is_graph_level() and task.is_classification():  # graph_classification
                graph_idx = self.element_idx

                edge_index = gen_dataset.dataset[graph_idx].edge_index
                y = gen_dataset.dataset[graph_idx].y
                x = gen_dataset.dataset[graph_idx].x

                model = model_manager.gnn
                model.eval()

                first_layer_name, layer = next(model.named_children())
                apply_decorator_to_graph_layers(model)

                budget = int(self.epsilon * edge_index.size(1))

                # TODO Some convolutions use the add_self_loops parameter. It is not possible to work with loops,
                #  because even if you remove a loop during the method, it will be used in the model architecture.
                # if layer.add_self_loops is True:
                #     pass

                perturbed_edges = edge_index.clone()
                self_loops_part_size = x.size(0)

                for _ in tqdm(range(budget)):
                    out = model(x, perturbed_edges)
                    loss = -model_manager.loss_function(*move_to_same_device(out, y))
                    loss.backward()

                    grad = layer.message_gradients[first_layer_name]
                    if self.grad_aggr_type == 'mean':
                        grad = torch.mean(grad, dim=1)
                    else:
                        raise ValueError(f"Unsupported grad_aggr_type: {self.grad_aggr_type}")

                    if grad.size(0) != perturbed_edges.size(1):  # model use add_self_loops
                        grad = grad[:-self_loops_part_size]

                    max_index = torch.argmax(grad)
                    perturbed_edges = torch.cat((perturbed_edges[:, :max_index], perturbed_edges[:, max_index + 1:]),
                                                dim=1)

                # from torch_geometric.data import Data
                # self.attack_res_misha = Data(x=x, edge_index=perturbed_edges, y=y)
                set_a = set(map(tuple, edge_index.T.tolist()))
                set_b = set(map(tuple, perturbed_edges.T.tolist()))

                diff_a = list(set_a - set_b)
                diff_b = list(set_b - set_a)

                gni = GlobalNodeIndexer(gen_dataset.dataset)
                diff_a = [tuple(gni.to_global(graph_idx, node) for node in edge) for edge in diff_a]
                diff_b = [tuple(gni.to_global(graph_idx, node) for node in edge) for edge in diff_b]

                self.attack_diff.remove_edges(diff_a)
                self.attack_diff.add_edges(diff_b)

            elif task.is_node_level() and task.is_classification():  # node_classification
                node_idx = self.element_idx

                edge_index = gen_dataset.data.edge_index
                y = gen_dataset.data.y
                x = gen_dataset.data.x

                model = model_manager.gnn
                model.eval()
                num_hops = model.n_layers

                print('node_idx', node_idx)
                subset, edge_index_subset, inv, edge_mask = k_hop_subgraph(node_idx=node_idx,
                                                                           num_hops=num_hops,
                                                                           edge_index=edge_index,
                                                                           relabel_nodes=True,
                                                                           directed=False)
                node_idx_remap = torch.where(subset == node_idx)[0].item()
                y = y.clone()
                y = y[subset]
                x = x.clone()
                x = x[subset]

                first_layer_name, layer = next(model.named_children())
                apply_decorator_to_graph_layers(model)

                budget = int(self.epsilon * edge_index_subset.size(1))

                # TODO Some convolutions use the add_self_loops parameter. It is not possible to work with loops,
                #  because even if you remove a loop during the method, it will be used in the model architecture.
                # if layer.add_self_loops is True:
                #     pass

                perturbed_edges = edge_index_subset.clone()
                self_loops_part_size = subset.size(0)

                for _ in tqdm(range(budget)):
                    out = model(x, perturbed_edges)
                    loss = -model_manager.loss_function(*move_to_same_device(out[node_idx_remap], y[node_idx_remap]))
                    loss.backward()

                    grad = layer.message_gradients[first_layer_name]
                    if self.grad_aggr_type == 'mean':
                        grad = torch.mean(grad, dim=1)
                    else:
                        raise ValueError(f"Unsupported grad_aggr_type: {self.grad_aggr_type}")

                    if grad.size(0) != perturbed_edges.size(1):  # model use add_self_loops
                        grad = grad[:-self_loops_part_size]

                    max_index = torch.argmax(grad)
                    perturbed_edges = torch.cat((perturbed_edges[:, :max_index], perturbed_edges[:, max_index + 1:]),
                                                dim=1)

                # Update dataset
                edges_to_keep = edge_index[:, ~edge_mask]
                updated_edge_index = torch.cat([edges_to_keep, perturbed_edges], dim=1)
                gen_dataset.data.edge_index = updated_edge_index
                # self.attack_res_misha = gen_dataset

                set_a = set(map(tuple, edge_index.T.tolist()))
                set_b = set(map(tuple, updated_edge_index.T.tolist()))

                diff_a = list(set_a - set_b)
                diff_b = list(set_b - set_a)

                self.attack_diff.remove_edges(diff_a)
                self.attack_diff.add_edges(diff_b)

            else:
                pass

        # return gen_dataset

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        return self.attack_diff


class PGDAttacker(
    EvasionAttacker
):
    name = "PGD"

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        """ Availability check for the given dataset and model manager. """
        return True

    def __init__(
            self,
            is_feature_attack: bool = False,
            element_idx: int = 0,
            epsilon: float = 10,
            learning_rate: float = 0.001,
            num_iterations: int = 100,
            random_sampling_num_trials: int = 100,
    ):

        super().__init__()
        self.is_feature_attack = is_feature_attack  # feature / structure
        self.element_idx = element_idx
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_sampling_num_trials = random_sampling_num_trials
        self.attack_diff = GraphModificationArtifact()
        self.attack_res = None

        # Process epsilon depending on attack type
        if not is_feature_attack:
            # For structure attack: treat epsilon as fraction or float budget → convert to int
            self.epsilon = int(epsilon)
        else:
            self.epsilon = epsilon  # For feature attack: leave as float

    @staticmethod
    def get_attack_loss(model_manager):
        """
        Returns a loss function that computes -loss_fn(pred, target). Lower loss value means better attack.
        """
        def attack_loss(output, target):
            return -model_manager.loss_function(output, target)
        return attack_loss

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor,
            task_type: str = None,
    ) -> None:
        if task_type is None:
            task_type = gen_dataset.is_multi()
        elif task_type == "multiple-graphs":
            task_type = True
        else:
            task_type = False

        model = model_manager.gnn
        model.eval()
        attack_loss = self.get_attack_loss(model_manager)

        """
        if self.is_feature_attack:
            orig_x = x.clone()
            x.requires_grad = True

            for _ in tqdm(range(self.num_iterations)):
                if x.grad is not None:
                    x.grad.zero_()
                if task_type:
                    out = model(x, edge_index)
                    loss = attack_loss(out, y)
                else:
                    out = model(x, edge_index_subset)
                    loss = attack_loss(out[node_idx_remap], y[node_idx_remap])
                loss.backward()
                with torch.no_grad():
                    x -= self.learning_rate * x.grad.sign()
                    x.copy_(torch.max(torch.min(x, orig_x + self.epsilon), orig_x - self.epsilon))
                    x.copy_(torch.clamp(x, -self.epsilon, self.epsilon))

            # Save changes in self.attack_res
            if task_type:
                self.attack_res = Data(x=x, edge_index=edge_index, y=y)
            else:
                gen_dataset.data.x[subset] = x
                self.attack_res = gen_dataset

            # Save changes
            if task_type:
                gni = GlobalNodeIndexer(gen_dataset.dataset)
                for node_idx in range(x.size(0)):
                    for feature_idx in range(x.size(1)):
                        self.attack_diff.change_node_feature(gni.to_global(graph_idx, node_idx), feature_idx,
                                                             x[node_idx][feature_idx].detach().cpu().item())
            else:
                for remap_idx, node_idx in enumerate(subset.detach().cpu()):
                    for feature_idx in range(x.size(1)):
                        self.attack_diff.change_node_feature(node_idx, feature_idx,
                                                             x[remap_idx][feature_idx].detach().cpu().item())"""

        if self.is_feature_attack:
            device = gen_dataset.data.x.device

            if task_type:
                graph_idxs = mask_tensor.nonzero(as_tuple=True)[0]  # LongTensor индексов графов

                selected_graphs = [gen_dataset.dataset[i].clone() for i in graph_idxs.tolist()]

                batch = Batch.from_data_list(selected_graphs).to(device)

                x = batch.x.clone()
                edge_index = batch.edge_index
                y = batch.y
                batch_vec = batch.batch

                orig_x = x.clone()
                x.requires_grad_(True)

                for _ in tqdm(range(self.num_iterations)):
                    if x.grad is not None:
                        x.grad.zero_()

                    out = model(x, edge_index, batch_vec)
                    loss = attack_loss(out, y)
                    loss.backward()

                    with torch.no_grad():
                        x -= self.learning_rate * x.grad.sign()
                        x.copy_(torch.max(torch.min(x, orig_x + self.epsilon), orig_x - self.epsilon))
                        x.copy_(torch.clamp(x, -self.epsilon, self.epsilon))

                ptr = batch.ptr
                attacked = []
                for j, gi in enumerate(graph_idxs.tolist()):
                    start = int(ptr[j])
                    end = int(ptr[j + 1])

                    attacked.append({
                        "graph_idx": gi,
                        "x": x[start:end].detach().clone(),
                        "edge_index": selected_graphs[j].edge_index
                    })
                self.attack_res = attacked

            else:
                x = gen_dataset.data.x.clone()
                edge_index = gen_dataset.data.edge_index.clone()
                y = gen_dataset.data.y.clone()
                # node_idxs = mask_tensor.nonzero(as_tuple=True)[0].tolist()

                orig_x = x.clone()
                x.requires_grad = True

                for _ in tqdm(range(self.num_iterations)):
                    if x.grad is not None:
                        x.grad.zero_()
                    out = model(x, edge_index)
                    loss = attack_loss(out[mask_tensor], y[mask_tensor])
                    loss.backward()
                    with torch.no_grad():
                        x -= self.learning_rate * x.grad.sign()
                        x.copy_(torch.max(torch.min(x, orig_x + self.epsilon), orig_x - self.epsilon))
                        x.copy_(torch.clamp(x, -self.epsilon, self.epsilon))

                self.attack_res = Data(x=x, edge_index=edge_index, y=y)
                gen_dataset.data.x = x  # FIXME tmp

        else:  # structure attack
            if task_type:
                graph_idx = self.element_idx
                x = gen_dataset.dataset[graph_idx].x.clone()
                edge_index = gen_dataset.dataset[graph_idx].edge_index.clone()
                y = gen_dataset.dataset[graph_idx].y.clone()
            else:
                node_idx = self.element_idx
                x = gen_dataset.data.x.clone()
                edge_index = gen_dataset.data.edge_index.clone()
                y = gen_dataset.data.y.clone()

                num_hops = model.n_layers
                subset, edge_index_subset, inv, edge_mask_k_hop = k_hop_subgraph(
                    node_idx=node_idx,
                    num_hops=num_hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    directed=False
                )
                node_idx_remap = torch.where(subset == node_idx)[0].item()
                x = x[subset]
                y = y[subset]

            if task_type:
                num_edges = edge_index.size(1)
            else:
                num_edges = edge_index_subset.size(1)
            wrapped_model = EdgeMaskingWrapper(copy.deepcopy(model), num_edges=num_edges)
            edge_mask = wrapped_model.edge_mask  # optimized mask

            for i in tqdm(range(self.num_iterations)):
                if task_type:
                    out = wrapped_model(x, edge_index)
                    loss = attack_loss(out, y)
                else:
                    out = wrapped_model(x, edge_index_subset)
                    loss = attack_loss(out[node_idx_remap], y[node_idx_remap])
                loss.backward()
                with torch.no_grad():
                    grad = edge_mask.grad
                    edge_mask -= self.learning_rate * grad
                    edge_mask.clamp_(0, 1)

            # Random sampling from probabilistic to binary topology perturbation
            if task_type:
                edge_index_for_random_sampling = edge_index
                target_idx = None
            else:
                edge_index_for_random_sampling = edge_index_subset
                target_idx = node_idx_remap

            best_binary_mask = random_sampling(
                edge_mask=edge_mask,
                budget=self.epsilon,
                num_trials=self.random_sampling_num_trials,
                model=model,
                x=x,
                edge_index=edge_index_for_random_sampling,
                y=y,
                attack_loss=attack_loss,
                target_idx=target_idx
            )

            # Save changes in self.attack_res
            if task_type:
                # edges_to_delete = edge_index[:, best_binary_mask].T.tolist()
                edges_to_keep = edge_index[:, ~best_binary_mask]
                # gen_dataset.data.edge_index = edges_to_keep
                self.attack_res = Data(x=x, edge_index=edges_to_keep, y=y)
            else:
                edges_to_keep = edge_index[:, ~edge_mask_k_hop]
                perturbed_edges = edge_index_subset[:, best_binary_mask]
                updated_edge_index = torch.cat([edges_to_keep, perturbed_edges], dim=1)

                gen_dataset.data.edge_index = updated_edge_index
                self.attack_res = gen_dataset

            # Save changes
            if task_type:
                gni = GlobalNodeIndexer(gen_dataset.dataset)
                edges_to_delete = edge_index[:, best_binary_mask].T.tolist()
                for edge in edges_to_delete:
                    self.attack_diff.remove_edge(gni.to_global(graph_idx, edge[0]), gni.to_global(graph_idx, edge[1]))
            else:
                subset_edge_indices = edge_mask_k_hop.nonzero(as_tuple=True)[0]
                edges_to_delete_indices = subset_edge_indices[best_binary_mask]
                for idx in edges_to_delete_indices:
                    edge = edge_index[:, idx].tolist()
                    self.attack_diff.remove_edge(edge[0], edge[1])

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        return self.attack_diff


class NettackAttacker(
    EvasionAttacker
):
    name = "Nettack"

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        """ Availability check for the given dataset and model manager. """
        if gen_dataset.is_multi():
            return False
        else:
            x = gen_dataset.data.x
            is_binary = torch.all((x == 0) | (x == 1)).item()
            return is_binary

    def __init__(
            self,
            node_idx: int = 0,
            budget: Union[int, None] = None,
            perturb_features: bool = True,
            perturb_structure: bool = True,
            direct: bool = True,
            depth: Union[int, None] = 2,
            delta_cutoff: float = 0.004,
            surrogate_train_ratio: float = 0.8,
            surrogate_epochs: int = 200
    ):
        super().__init__()
        self.attack_diff = None
        self.node_idx = node_idx
        self.budget = budget
        self.perturb_features = perturb_features
        self.perturb_structure = perturb_structure
        self.direct = direct
        self.depth = depth
        self.delta_cutoff = delta_cutoff
        self.surrogate_train_ratio = surrogate_train_ratio
        self.surrogate_epochs = surrogate_epochs
        self.attack_diff = GraphModificationArtifact()

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ):
        data = gen_dataset.data
        x, edge_index, y = move_to_same_device(data.x, data.edge_index, data.y, device=torch.device('cpu'))

        num_classes = y.max().item() + 1
        surrogate = NettackSurrogate(in_channels=x.size(1), out_channels=num_classes).to(x.device)
        surrogate.train_model(x, edge_index, y, train_ratio=self.surrogate_train_ratio, epochs=self.surrogate_epochs)
        # surrogate.evaluate(x, edge_index, y)

        attacker = NettackAttack(
            real_class=data.y[self.node_idx].item(),
            gnn_model=model_manager.gnn,
            model=surrogate,
            x=x,
            edge_index=edge_index,
            num_classes=num_classes,
            target_node=self.node_idx,
            attack_diff=self.attack_diff,
            direct=self.direct,
            depth=self.depth,
            delta_cutoff=self.delta_cutoff
        )

        mode = "both"
        if self.perturb_features and not self.perturb_structure:
            mode = "feature"
        elif self.perturb_structure and not self.perturb_features:
            mode = "structure"

        # logits_before = surrogate.forward(edge_index, x)[self.node_idx]
        # pred_before = logits_before.argmax().item()
        # prob_before = torch.softmax(logits_before, dim=0)[pred_before].item()
        # print(f"Surrogate prediction before attack: {pred_before} (confidence: {prob_before:.4f})")

        attacker.attack(budget=self.budget, mode=mode)

        # logits_after = surrogate.forward(attacker.edge_index, attacker.x)[self.node_idx]
        # pred_after = logits_after.argmax().item()
        # prob_after = torch.softmax(logits_after, dim=0)[pred_after].item()
        # print(f"Surrogate prediction after attack: {pred_after} (confidence: {prob_after:.4f})")
        # gen_dataset.data.x = attacker.x
        # gen_dataset.data.edge_index = attacker.edge_index
        # return attacker.x, attacker.edge_index

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        return self.attack_diff


class ReWattAttacker(
    EvasionAttacker
):
    """
    ReWatt: Reinforcement Learning-based Edge Rewiring Attack on GNNs.
    """

    name = "ReWatt"

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        """ Availability check for the given dataset and model manager. """
        if gen_dataset.is_multi():
            return True
        else:
            return len(model_manager.gnn.embedding_levels_by_layers) - 2 >= 0

    def __init__(
            self,
            element_idx: int = 0,
            eps: int = 0.1,
            epochs: int = 100,
            mlp_hidden: int = 16,
            h_method: str = 'sum',
            pooling_method: str = 'mean',
    ):
        """
        Initialize the ReWattAttacker.

        :param element_idx: Index of the node (for node classification) or graph (for graph classification) to attack.
        :type element_idx: int

        :param eps: Fraction (0 < eps <= 1) of total edges to be considered for rewiring (defines the attack budget).
        :type eps: float

        :param epochs: Number of training epochs for the reinforcement learning agent.
        :type epochs: int

        :param mlp_hidden: Hidden dimension size for the MLPs used in the policy network.
        :type mlp_hidden: int

        :param h_method: Method used to combine node embeddings into edge representations. Options: 'sum', 'mul', 'max'.
        :type h_method: str

        :param pooling_method: Method for aggregating node embeddings into a graph-level representation. Options: 'mean', 'max'.
        :type pooling_method: str

        :return: None
        :rtype: None
        """
        super().__init__()
        self.element_idx = element_idx
        self.eps = eps
        self.epochs = epochs
        self.mlp_hidden = mlp_hidden
        self.h_method = h_method
        self.pooling_method = pooling_method

        self.attack_diff = GraphModificationArtifact()
        self.my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ):
        """
        Executes the ReWatt attack on the given graph data and GNN model.

        :param model_manager: model manager.
        :type model_manager: Type

        :param gen_dataset: general dataset.
        :type gen_dataset: GeneralDataset

        :param mask_tensor: desc.
        :type mask_tensor: torch.Tensor

        :return: None.
        :rtype: None
        """
        model = model_manager.gnn
        model.eval()

        if gen_dataset.is_multi():
            graph_idx = self.element_idx
            node_idx=None
            edge_index = gen_dataset.dataset[graph_idx].edge_index.to(self.my_device)
            y = gen_dataset.dataset[graph_idx].y.squeeze().to(self.my_device)
            x = gen_dataset.dataset[graph_idx].x.to(self.my_device)
            y_prob = torch.softmax(model(x, edge_index), dim=1).squeeze().max().item()
            penultimate_layer_embeddings_idx = model.embedding_levels_by_layers.index('g') - 1
            penultimate_layer_embeddings_dim = (
                model.get_all_layer_embeddings(x, edge_index)[penultimate_layer_embeddings_idx].size(1))
        else:
            node_idx = self.element_idx
            y = gen_dataset.data.y[node_idx].to(self.my_device)
            x = gen_dataset.data.x.to(self.my_device)
            edge_index = gen_dataset.data.edge_index.to(self.my_device)
            y_prob = torch.softmax(model(x, edge_index)[node_idx], dim=0).max().item()
            penultimate_layer_embeddings_idx = len(model.embedding_levels_by_layers) - 2
            penultimate_layer_embeddings_dim = (
                model.get_all_layer_embeddings(x, edge_index)[penultimate_layer_embeddings_idx].size(1))

        # the attack makes sense when the model's prediction is correct !!!
        # we use y_prob in the attack because in case the attack fails to change the class, we have saved
        # the state of the graph that most reduces the probability of a correct prediction.
        initial_graph_state = GraphState(x, edge_index, y, y_prob, device=self.my_device)
        env = GraphEnvironment(model, initial_graph_state, eps=self.eps, node_idx=node_idx)

        policy = ReWattPolicyNet(
            gnn_model=model,
            penultimate_layer_embeddings_dim=penultimate_layer_embeddings_dim,
            penultimate_layer_embeddings_idx=penultimate_layer_embeddings_idx,
            node_idx=node_idx,
            mlp_hidden=self.mlp_hidden,
            h_method=self.h_method,
            pooling_method=self.pooling_method,
            device=self.my_device
        )

        agent = ReWattAgent(policy, env, lr=1e-3, gamma=0.99)
        attacked_graph = agent.train(epochs=self.epochs)

        set_a = set(map(tuple, edge_index.T.tolist()))
        set_b = set(map(tuple, attacked_graph.edge_index.T.tolist()))

        diff_a = list(set_a - set_b)
        diff_b = list(set_b - set_a)

        if gen_dataset.is_multi():
            gni = GlobalNodeIndexer(gen_dataset.dataset)
            diff_a = [tuple(gni.to_global(graph_idx, node) for node in edge) for edge in diff_a]
            diff_b = [tuple(gni.to_global(graph_idx, node) for node in edge) for edge in diff_b]

        self.attack_diff.remove_edges(diff_a)
        self.attack_diff.add_edges(diff_b)

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        return self.attack_diff
