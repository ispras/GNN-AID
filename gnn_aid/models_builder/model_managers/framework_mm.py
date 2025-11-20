import json
from math import ceil
from pathlib import Path
from typing import Union, List, cast, Iterable

import torch
from torch import tensor
from torch.cuda import is_available
from torch.nn.utils import clip_grad_norm
from torch_geometric.loader import NeighborLoader, DataLoader, LinkNeighborLoader
from torch_geometric.utils import negative_sampling

from gnn_aid.aux import Declare
from gnn_aid.aux.utils import OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH, \
    FRAMEWORK_PARAMETERS_PATH, move_to_same_device
from gnn_aid.data_structures import Task, GraphModificationArtifact
from gnn_aid.data_structures.configs import ConfigPattern, CONFIG_OBJ
from gnn_aid.datasets import GeneralDataset
from . import GNNModelManager
from gnn_aid.models_builder.models_utils import Metric


class FrameworkGNNModelManager(GNNModelManager):
    """
    GNN model control class. Have methods: train_model, save_model, load_model
    """
    gnn: torch.nn.Module
    additional_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_config_class": "Config",
                "_class_name": "Adam",
                "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            },
            # FUNCTIONS_PARAMETERS_PATH,
            "loss_function": {
                "_config_class": "Config",
                "_class_name": "NLLLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
        }
    )
    """
    Args not listed in meta-info (thus not shown at frontend) to be added at initialization.
    
    optimizer: optimizer class name and params, all supported classes are described in optimizers_parameters.json file
    batch: int, batch to train
    loss_function: loss_function class name and params, all supported classes are described 
    in functions_parameters.json file
    clip: float, clip to train. If not None call clip_grad_norm
    mask_features: list of the names of the features to be masked. For example, 
    to prevent leakage of the response during training.
    """

    def __init__(self, gnn: torch.nn.Module = None,
                 dataset_path: Union[str, Path] = None,
                 **kwargs
                 ):
        """
        :param gnn: graph neural network model based on the GNNConstructor class
        :param manager_config:
        :param modification:
        :param dataset_path: int, the number of epochs the model actually was trained
        :param epochs: int, the number of epochs the model actually was trained
        :param kwargs: kwargs for GNNModelManager
        """

        # TODO Kirill, add train_test_split in default parameters gnnMM
        super().__init__(**kwargs)
        # Fulfill absent fields from default configs
        with open(FRAMEWORK_PARAMETERS_PATH, 'r') as f:
            params = json.load(f)
            class_name = type(self).__name__
            if class_name in params:
                self.manager_config = ConfigPattern(
                    _config_class="ModelManagerConfig",
                    _config_kwargs={k: v[2] for k, v in params[class_name].items()},
                ).merge(self.manager_config)

        # Add fields from additional config
        self.manager_config = self.additional_config.merge(self.manager_config)

        self.gnn = gnn

        if self.modification.epochs is None:
            self.modification.epochs = 0
        self.optimizer = None
        self.loss_function = None

        self.batch = getattr(self.manager_config, CONFIG_OBJ).batch
        self.clip = getattr(self.manager_config, CONFIG_OBJ).clip
        self.mask_features = getattr(self.manager_config, CONFIG_OBJ).mask_features

        self.dataset_path = dataset_path

        if self.gnn is not None:
            self.init()

    def init(
            self
    ) -> None:
        """
        Initialize optimizer and loss function.
        """
        if self.gnn is None:
            raise Exception("FrameworkGNNModelManager need GNN, now GNN is None")

        # QUE Kirill, can we make this better
        if "optimizer" in getattr(self.manager_config, CONFIG_OBJ):
            self.optimizer = getattr(self.manager_config, CONFIG_OBJ).optimizer.create_obj(params=self.gnn.parameters())
            # self.optimizer = getattr(self.manager_config, CONFIG_OBJ).optimizer.create_obj()

        if "loss_function" in getattr(self.manager_config, CONFIG_OBJ):
            self.loss_function = getattr(self.manager_config, CONFIG_OBJ).loss_function.create_obj()

    def train_complete(
            self,
            gen_dataset: GeneralDataset,
            steps: int = None,
            pbar: Union['tqdm', None] = None,
            metrics: Union[List[Metric], Metric] = None,
            **kwargs
    ) -> None:
        for _ in range(steps):
            self.before_epoch(gen_dataset)
            print("epoch", self.modification.epochs)
            train_loss = self.train_1_step(gen_dataset)
            self.after_epoch(gen_dataset)
            early_stopping_flag = self.early_stopping(train_loss=train_loss, gen_dataset=gen_dataset,
                                                      metrics=metrics, steps=steps)
            if self.socket:
                self.report_results(train_loss=train_loss, gen_dataset=gen_dataset,
                                    metrics=metrics)
            pbar.update(1)
            if early_stopping_flag:
                break

    def early_stopping(
            self,
            train_loss,
            gen_dataset: GeneralDataset,
            metrics: Union[List[Metric], Metric],
            steps: int
    ) -> bool:
        return False

    def train_1_step(
            self,
            gen_dataset: GeneralDataset
    ) -> List[Union[float, int]]:
        task_type = gen_dataset.dataset_var_config.task
        if task_type in [Task.NODE_CLASSIFICATION, Task.NODE_REGRESSION]:
            # FIXME Kirill, add data_x_copy mask
            loader = cast(
                Iterable,
                NeighborLoader(
                    gen_dataset.data,
                    num_neighbors=[-1], input_nodes=gen_dataset.train_mask,
                    batch_size=self.batch, shuffle=True
                )
            )
        elif task_type in [Task.GRAPH_CLASSIFICATION, Task.GRAPH_REGRESSION]:
            train_dataset = gen_dataset.dataset.index_select(gen_dataset.train_mask)
            loader = cast(
                Iterable,
                DataLoader(
                    train_dataset, batch_size=self.batch, shuffle=True
                )
            )
        elif task_type == Task.EDGE_PREDICTION:
            edge_label_index = getattr(gen_dataset, 'edge_label_index', None)
            if edge_label_index is None:
                raise ValueError("data.edge_label_index is out")

            train_mask = getattr(gen_dataset, 'train_mask', None)
            if train_mask is None:
                raise ValueError("data.train_mask is out")

            pos_edge_index = edge_label_index[:, train_mask]
            pos_label = torch.ones(pos_edge_index.size(1), dtype=torch.long, device=gen_dataset.dataset.edge_index.device)

            # TODO num_neg_samples as MM parameter
            neg_edge_index = negative_sampling(
                edge_index=gen_dataset.data.edge_index,
                num_nodes=gen_dataset.data.num_nodes,
                num_neg_samples=pos_edge_index.size(1),
                method='sparse'
            )
            neg_label = torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=gen_dataset.dataset.edge_index.device)

            device = gen_dataset.dataset.edge_index.device
            pos_edge_index = pos_edge_index.to(device)
            neg_edge_index = neg_edge_index.to(device)
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat([pos_label, neg_label], dim=0)

            train_data = gen_dataset.data.clone()
            train_data.edge_label_index = edge_label_index
            train_data.edge_label = edge_label

            loader = cast(
                Iterable,
                LinkNeighborLoader(
                    data=train_data,
                    num_neighbors=[-1],
                    batch_size=self.batch,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    shuffle=True,
                )
            )

        else:
            raise ValueError(f"Unsupported task type {task_type}")
        loss = 0
        for batch in loader:
            self.before_batch(batch)
            loss += self.train_on_batch_full(batch, task_type)
            self.after_batch(batch)
        print("loss %.8f" % loss)
        self.modification.epochs += 1
        self.gnn.eval()
        return loss.cpu().detach().numpy().tolist()

    def train_on_batch_full(
            self,
            batch,
            task_type: Task
    ) -> torch.Tensor:
        if self.mi_defender and self.mi_defense_flag:
            self.mi_defender.pre_batch()
        if self.evasion_defender and self.evasion_defense_flag:
            self.evasion_defender.pre_batch(model_manager=self, batch=batch, task_type=task_type)
        loss = self.train_on_batch(batch=batch, task_type=task_type)
        mi_defender_dict = None
        if self.mi_defender and self.mi_defense_flag:
            # TODO pass decoder
            mi_defender_dict = self.mi_defender.post_batch(model_manager=self, batch=batch)
        if mi_defender_dict and "loss" in mi_defender_dict:
            loss = mi_defender_dict["loss"]
        evasion_defender_dict = None
        if self.evasion_defender and self.evasion_defense_flag:
            # TODO pass decoder
            evasion_defender_dict = self.evasion_defender.post_batch(
                model_manager=self, batch=batch, loss=loss,
            )
        if evasion_defender_dict and "loss" in evasion_defender_dict:
            loss = evasion_defender_dict["loss"]
        loss = self.optimizer_step(loss=loss)
        return loss

    def optimizer_step(
            self,
            loss: torch.Tensor
    ) -> torch.Tensor:
        loss.backward()
        self.optimizer.step()
        return loss

    def train_on_batch(
            self,
            batch,
            task_type: Task = None
    ) -> torch.Tensor:
        loss = None
        if hasattr(batch, "edge_weight"):
            weight = batch.edge_weight
        else:
            weight = None
        if task_type in [Task.NODE_CLASSIFICATION, Task.NODE_REGRESSION]:
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, weight)

            loss = self.loss_function(*move_to_same_device(logits[:batch.batch_size], batch.y[:batch.batch_size]))
            if self.clip is not None:
                clip_grad_norm(self.gnn.parameters(), self.clip)
            self.optimizer.zero_grad()
        elif task_type in [Task.GRAPH_CLASSIFICATION, Task.GRAPH_REGRESSION]:
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, batch.batch, weight)
            loss = self.loss_function(*move_to_same_device(logits, batch.y))
        elif task_type == Task.EDGE_PREDICTION:
            self.optimizer.zero_grad()
            device = batch.x.device

            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_label_index = batch.edge_label_index.to(device)
            edge_label = batch.edge_label.to(device).float()
            node_embeddings = self.gnn(x, edge_index, weight=weight if 'weight' in locals() else None)

            src = node_embeddings[edge_label_index[0]]
            dst = node_embeddings[edge_label_index[1]]
            # TODO make decoder - returns 2 numbers
            out = (src * dst).sum(dim=-1)

            # FIXME loss must be for edge prediction
            loss = self.loss_function(out, edge_label)
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        return loss

    def get_name(
            self,
            **kwargs
    ) -> str:
        json_str = super().get_name()
        return json_str

    def load_model(
            self,
            path: Union[str, Path, None] = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Load model from torch save format

        :param path: path to load the model. By default, the path is compiled based on the global
         class variables
        """
        if not is_available():
            self.gnn.load_state_dict(torch.load(path, map_location=torch.device('cpu'), ))
            # self.gnn = torch.load(path, map_location=torch.device('cpu'))
        else:
            self.gnn.load_state_dict(torch.load(path, ))
            # self.gnn = torch.load(path)
        if self.optimizer is None:
            self.init()
        return self.gnn

    def save_model(
            self,
            path: Union[str, Path] = None
    ) -> None:
        """
        Save the model in torch format

        :param path: path to save the model. By default,
         the path is compiled based on the global class variables
        """
        torch.save(self.gnn.state_dict(), path)

    def report_results(
            self,
            train_loss,
            gen_dataset: GeneralDataset,
            metrics: List[Metric]
    ) -> None:
        metrics_values = self.evaluate_model(gen_dataset=gen_dataset, metrics=metrics)
        self.compute_stats_data(gen_dataset, predictions=True, logits=True)
        self.send_epoch_results(
            metrics_values=metrics_values,
            stats_data={k: gen_dataset.visible_part.filter(v)
                        for k, v in self.stats_data.items()},
            weights={"weights": self.gnn.get_weights()}, loss=train_loss)

    def train_model(
            self,
            gen_dataset: GeneralDataset,
            save_model_flag: bool = True,
            mode: Union[str, None] = None,
            steps=None,
            metrics: List[Metric] = None,
            socket: 'SocketConnect' = None
    ) -> Union[str, Path]:
        """
        Convenient train method.

        :param gen_dataset: dataset in torch_geometric data format for train
        :param save_model_flag: if need save model after train. Default save_model_flag=True
        :param mode: '1 step' or 'full' or None (choose automatically)
        :param steps: train specific number of epochs, if None - all of them
        :param metrics: list of metrics to measure at each step or at the end of training
        :param socket: socket to use for sending data to frontend
        """
        from gnn_aid.explainers.explainer import ProgressBar
        gen_dataset = self.load_or_execute_poisoning_attack(
            gen_dataset=gen_dataset
        )
        gen_dataset = self.load_or_execute_poisoning_defense(
            gen_dataset=gen_dataset
        )

        self.socket = socket
        pbar = ProgressBar(self.socket, "mt")

        # Assure we call here from subclass
        assert issubclass(type(self), GNNModelManager)

        assert mode in ['1_step', 'full', None]
        # TODO Kirill what is this? Outdated?
        # has_complete = self.train_complete != super(type(self), self).train_complete
        # assert has_complete
        do_1_step = True

        try:
            if do_1_step:
                assert steps > 0
                pbar.total = self.modification.epochs + steps
                pbar.n = self.modification.epochs
                pbar.update(0)
                self.train_complete(gen_dataset=gen_dataset, steps=steps,
                                    pbar=pbar, metrics=metrics)
                pbar.close()
                self.send_data("mt", {"status": "OK"})

            else:
                raise Exception

            if save_model_flag:
                return self.save_model_executor()

        except Exception as e:
            self.send_data("mt", {"status": "FAILED"})
            raise e
        finally:
            self.socket = None

    def run_model(
            self,
            gen_dataset: GeneralDataset,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
            out: str = 'answers'
    ) -> torch.Tensor:
        """
        Run the model on a part of dataset specified with a mask.

        :param gen_dataset: wrapper over the dataset, stores the dataset and all meta-information about the dataset
        :param mask: 'train', 'val', 'test', 'all' -- part of the dataset on which the output will be obtained
        :param out: 'answers', 'predictions', 'logits' -- what output format will be calculated,
         availability depends on which methods have been overridden in the parent class
        :return:
        """
        try:
            mask = {
                'train': gen_dataset.train_mask,
                'val': gen_dataset.val_mask,
                'test': gen_dataset.test_mask,
                'all': tensor([True] * len(gen_dataset.labels)),
            }[mask]
        except KeyError:
            assert isinstance(mask, torch.Tensor)

        run_func = {
            'answers': self.gnn.get_answer,
            'predictions': self.gnn.get_predictions,
            'logits': self.gnn.__call__,
        }[out]
        self.gnn.eval()
        with torch.no_grad():  # Turn off gradients computation
            if gen_dataset.is_multi():
                dataset = gen_dataset.dataset
                part_loader = cast(
                    Iterable,
                    DataLoader(
                        dataset.index_select(mask),
                        batch_size=self.batch,
                        shuffle=False
                    )
                )
                full_out = torch.empty(0, device=gen_dataset.data.x.device)
                # y_true = torch.Tensor()
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                for data in part_loader:
                    # logits_batch = self.gnn(data.x, data.edge_index, data.batch)
                    # pred_batch = logits_batch.argmax(dim=1)
                    out = run_func(data.x, data.edge_index, data.batch)
                    full_out = torch.cat((move_to_same_device(full_out, out)))
                    # y_true = torch.cat((y_true, data.y))
            else:  # single-graph
                data = gen_dataset.data  # FIXME what if no data? use .get(0) ?
                ver_ind = [n for n, x in enumerate(mask) if x]
                mask_size = len(ver_ind)

                number_of_batches = ceil(mask_size / self.batch)
                # data_x_elem_len = data.x.size()[1]
                full_out = torch.empty(0, device=data.x.device)
                # features_mask_tensor = torch.full(size=data.x.size(), fill_value=True)

                for batch_ind in range(number_of_batches):
                    data_x_copy = torch.clone(data.x)
                    mask_copy = [False] * data.x.size()[0]

                    # features_mask_tensor_copy = torch.clone(features_mask_tensor)

                    for elem_ind in ver_ind[
                                    batch_ind * self.batch: (batch_ind + 1) * self.batch]:
                        if hasattr(self, 'mask_features'):
                            for feature in self.mask_features:
                                # features_mask_tensor_copy[elem_ind][gen_dataset.node_attr_slices[feature][0]:
                                #                                     gen_dataset.node_attr_slices[feature][1]] = False
                                data_x_copy[elem_ind][gen_dataset.node_attr_slices[feature][0]:
                                                      gen_dataset.node_attr_slices[feature][1]] = 0
                        # if self.gnn_mm.train_mask_flag:
                        #     data_x_copy[elem_ind] = torch.zeros(data_x_elem_len)
                        # y_true = torch.masked.masked_tensor(data.y, mask_tensor)
                        mask_copy[elem_ind] = True

                    # mask_x_tensot = torch.masked.masked_tensor(data.x, features_mask_tensor_copy)

                    # FIXME Kirill what to do if no optimizer, train_mask_flag, batch?
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                    # logits_batch = self.gnn(data_x_copy, data.edge_index)
                    # pred_batch = logits_batch.argmax(dim=1)
                    out = run_func(data_x_copy, data.edge_index)
                    full_out = torch.cat((move_to_same_device(full_out, out[mask_copy])))
                    # y_true = torch.cat((y_true, data.y[mask_copy]))

        return full_out

    def load_or_execute_poisoning_attack(
            self,
            gen_dataset: GeneralDataset,
            poison_attack_diff_file_path: Union[str, Path] = None,
    ) -> GeneralDataset:
        """
        Loads and applies poisoning attack artifacts to the dataset if available; otherwise, executes the attack
        and generates the necessary artifacts.

        Parameters
        ----------
        gen_dataset : GeneralDataset
            Object containing dataset data and configuration metadata.
        poison_attack_diff_file_path : str, optional
            Path to precomputed poisoning artifacts. If None, the default path from the dataset configuration is used.

        Returns
        -------
        GeneralDataset
            A modified dataset with the poisoning attack applied.
        """
        if poison_attack_diff_file_path is None:
            _, files_paths = Declare.models_path(self)
            poison_attack_diff_file_path = files_paths[3]
        if self.poison_attacker and self.poison_attack_flag:
            try:
                artifact = GraphModificationArtifact.from_json(poison_attack_diff_file_path)
                gen_dataset = gen_dataset.apply_modification(artifact=artifact)
            except Exception as e:
                print(f"An error occurred: {type(e).__name__} - {e}")
                loc = self.poison_attacker.attack(gen_dataset=gen_dataset)
                self.poison_attacker.dataset_diff()
                if loc is not None:
                    gen_dataset = loc
        return gen_dataset

    def load_or_execute_poisoning_defense(
            self,
            gen_dataset: GeneralDataset,
            poison_defense_diff_file_path: Union[str, Path] = None,
    ) -> GeneralDataset:
        """
        Loads and applies defense artifacts against poisoning attacks if available; otherwise, executes the defense
        method and generates the necessary artifacts.

        Parameters
        ----------
        gen_dataset : GeneralDataset
            Object containing dataset data and configuration metadata.
        poison_defense_diff_file_path : str, optional
            Path to precomputed defense artifacts. If None, the default path from the dataset configuration is used.

        Returns
        -------
        GeneralDataset
            A modified dataset with the poisoning defense applied.
        """
        if poison_defense_diff_file_path is None:
            _, files_paths = Declare.models_path(self)
            poison_defense_diff_file_path = files_paths[5]
        if self.poison_defender and self.poison_defense_flag:
            try:
                artifact = GraphModificationArtifact.from_json(poison_defense_diff_file_path)
                gen_dataset = gen_dataset.apply_modification(artifact=artifact)
            except Exception as e:
                print(f"An error occurred: {type(e).__name__} - {e}")
                loc = self.poison_defender.defense(gen_dataset=gen_dataset)
                self.poison_defender.dataset_diff()
                if loc is not None:
                    gen_dataset = loc
        return gen_dataset

    def evaluate_model(
            self,
            gen_dataset: GeneralDataset,
            metrics: Union[List[Metric], Metric, torch.Tensor]
    ) -> dict:
        """
        Compute metrics for a model result on a part of dataset specified by the metric mask.

        :param gen_dataset: wrapper over the dataset, stores the dataset and all meta-information about the dataset
        :param metrics: list of metrics to compute. metric based on class Metric
        :return: dict {metric -> value}
        """
        mask_metrics = {}
        for metric in metrics:
            mask = metric.mask
            if mask not in mask_metrics:
                mask_metrics[mask] = []
            mask_metrics[mask].append(metric)

        metrics_values = {}
        for mask, ms in mask_metrics.items():
            try:
                mask_tensor = {
                    'train': gen_dataset.train_mask.tolist(),
                    'val': gen_dataset.val_mask.tolist(),
                    'test': gen_dataset.test_mask.tolist(),
                    'all': [True] * len(gen_dataset.labels),
                }[mask]
            except KeyError:
                assert isinstance(mask, torch.Tensor)
                mask_tensor = mask
            if self.evasion_attacker and self.evasion_attack_flag:
                # TODO pass decoder
                self.call_evasion_attack(
                    gen_dataset=gen_dataset,
                    mask=mask,
                )
            metrics_values[mask] = {}
            y_pred = self.run_model(gen_dataset, mask=mask)
            y_true = gen_dataset.labels[mask_tensor]

            for metric in ms:
                metrics_values[mask][metric.name] = metric.compute(y_pred, y_true)
                # metrics_values[mask][metric.name] = MetricManager.compute(metric, y_pred, y_true)
        if self.mi_attacker and self.mi_attack_flag:
            # TODO pass decoder
            self.call_mi_attack(gen_dataset=gen_dataset, mask_tensor=mask, model=self.gnn)
        return metrics_values

    def call_evasion_attack(
            self,
            gen_dataset: GeneralDataset,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ):
        if self.evasion_attacker:
            try:
                mask_tensor = {
                    'train': gen_dataset.train_mask.tolist(),
                    'val': gen_dataset.val_mask.tolist(),
                    'test': gen_dataset.test_mask.tolist(),
                    'all': [True] * len(gen_dataset.labels),
                }[mask]
            except KeyError:
                assert isinstance(mask, torch.Tensor)
                mask_tensor = mask
            self.evasion_attacker.attack(
                model_manager=self,
                gen_dataset=gen_dataset,
                mask_tensor=mask_tensor
            )

    def call_mi_attack(
            self,
            gen_dataset: GeneralDataset,
            model: torch.nn.Module,
            mask_tensor: Union[str, List[bool], torch.Tensor] = 'test'
    ):
        if self.mi_attacker:
            self.mi_attacker.attack(gen_dataset=gen_dataset, model=model, mask_tensor=mask_tensor)

    def compute_stats_data(
            self,
            gen_dataset: GeneralDataset,
            predictions: bool = False,
            logits: bool = False
    ):
        """
        :param gen_dataset: wrapper over the dataset, stores the dataset
         and all meta-information about the dataset
        :param predictions: boolean flag that indicates the need to enter model predictions
         in the statistics for the front
        :param logits: boolean flag that indicates the need to enter model logits
         in the statistics for the front
        :return: dict with model weights. Also function can add in dict model predictions
         and logits
        """
        stats_data = {}

        # Stats: weights, logits, predictions
        if predictions:  # and hasattr(self.gnn, 'get_predictions'):
            predictions = self.run_model(gen_dataset, mask='all', out='predictions')
            stats_data["predictions"] = predictions.detach().cpu().tolist()
        if logits:  # and hasattr(self.gnn, 'forward'):
            logits = self.run_model(gen_dataset, mask='all', out='logits')
            stats_data["embeddings"] = logits.detach().cpu().tolist()

        # Note: we update all stats data at once because it can be requested from frontend during
        # the update
        self.stats_data = stats_data

    def send_data(
            self,
            block,
            msg,
            tag='model',
            obligate=True,
            socket=None
    ):
        """
        Send data to the frontend.

        :param socket:
        :param tag:
        :param block:
        :param msg: message as a json-convertible dict
        :param obligate: if you send a lot of updates of the same stuff, e.g. weights at each
         training step, set obligate=False to save traffic and actually send only the last one on
         the queue.
        :return: bool flag
        """
        socket = socket or self.socket
        if socket is None:
            return False
        socket.send(
            block=block,
            msg=msg,
            tag=tag,
            obligate=obligate
        )
        return True

    def send_epoch_results(
            self,
            metrics_values=None,
            stats_data=None,
            weights=None,
            loss=None,
            obligate=False,
            socket=None
    ):
        """
        Send updates to the frontend after a training epoch: epoch, metrics, logits, loss.

        :param weights:
        :param metrics_values: quality metrics (accuracy, F1)
        :param stats_data: model statistics (logits, predictions)
        :param loss: train loss
        """
        socket = socket or self.socket
        # Metrics values, epoch, loss
        if metrics_values:
            metrics_data = {"epochs": self.modification.epochs}
            if loss:
                metrics_data["loss"] = loss
            metrics_data["metrics_values"] = metrics_values
            self.send_data("mt", {"metrics": metrics_data}, tag='model_metrics', socket=socket)
        if weights:
            self.send_data("mt", weights, tag='model_weights', obligate=obligate, socket=socket)
        if stats_data:
            self.send_data("mt", stats_data, tag='model_stats', obligate=obligate, socket=socket)

    def load_train_test_split(
            self,
            gen_dataset: GeneralDataset
    ) -> GeneralDataset:
        path = self.model_path_info()
        path = path / 'train_test_split'
        gen_dataset.train_mask, gen_dataset.val_mask, gen_dataset.test_mask, _ = torch.load(path)[:]
        return gen_dataset
