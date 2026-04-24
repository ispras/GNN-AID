from pathlib import Path
from typing import Type, Union, List

import torch
from torch.cuda import is_available
from torch.nn.utils import clip_grad_norm

from gnn_aid.aux.utils import OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH, \
    move_to_same_device
from gnn_aid.data_structures import Task
from gnn_aid.data_structures.gen_config import CONFIG_OBJ, ConfigPattern
from gnn_aid.datasets import GeneralDataset
from gnn_aid.models_builder.models_utils import Metric
from . import FrameworkGNNModelManager


class ProtGNNModelManager(FrameworkGNNModelManager):
    """
    Model manager for ProtGNN training.

    Extends FrameworkGNNModelManager with prototype projection,
    cluster/separation losses, and early stopping based on validation accuracy.
    """
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
                "_class_name": "CrossEntropyLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
        }
    )

    def __init__(
            self,
            gnn: torch.nn.Module = None,
            dataset_path: Union[str, Path] = None,
            **kwargs
    ):
        """
        Args:
            gnn (torch.nn.Module): ProtGNN model with a ``prot_layer_name`` attribute.
                Default value: `None`.
            dataset_path (Union[str, Path]): Path to the dataset. Default value: `None`.
            **kwargs: Additional arguments forwarded to FrameworkGNNModelManager.
        """
        super().__init__(gnn=gnn, dataset_path=dataset_path, **kwargs)

        # Get prot layer and its params
        self.is_best = None
        self.cur_acc = None
        self.prot_layer = getattr(self.gnn, self.gnn.prot_layer_name)
        _config_obj = getattr(self.manager_config, CONFIG_OBJ)
        self.clst = _config_obj.clst
        self.sep = _config_obj.sep
        # lr = _config_obj.lr
        self.early_stopping_marker = _config_obj.early_stopping
        self.proj_epochs = _config_obj.proj_epochs
        self.warm_epoch = _config_obj.warm_epoch
        self.save_epoch = _config_obj.save_epoch
        self.save_thrsh = _config_obj.save_thrsh
        # TODO implement other MCTS args too
        # TODO MCTS args via static ?
        from gnn_aid.explainers.protgnn.MCTS import mcts_args
        mcts_args.min_atoms = _config_obj.mcts_min_atoms
        mcts_args.max_atoms = _config_obj.mcts_max_atoms
        self.prot_thrsh = _config_obj.prot_thrsh
        self.early_stop_count = 0
        self.gnn.best_prots = self.prot_layer.prototype_graphs
        self.best_acc = 0.0

    def save_model(
            self,
            path: Union[str, Path, None] = None
    ) -> None:
        """
        Save the model in torch format.
        By default, the path is compiled based on the global class variables
        """
        torch.save({"model_state_dict": self.gnn.state_dict(),
                    "best_prots": self.gnn.best_prots,
                    }, path)

    def load_model(
            self,
            path: Union[str, Path, None] = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Load model weights and best prototypes from a torch checkpoint.

        Args:
            path (Union[str, Path, None]): Path to the checkpoint. Default value: `None`.

        Returns:
            The GNN module with loaded weights.
        """
        if not is_available():
            checkpoint = torch.load(path, map_location=torch.device('cpu'), )
        else:
            checkpoint = torch.load(path)
        self.gnn.load_state_dict(checkpoint["model_state_dict"])
        self.gnn.best_prots = checkpoint["best_prots"]
        if self.optimizer is None:
            self.init()
        return self.gnn

    def train_on_batch(
            self,
            batch,
            task_type: Task
    ) -> torch.Tensor:
        """
        Compute the ProtGNN loss for one batch.

        Includes cross-entropy, cluster, separation, and sparsity losses.

        Args:
            batch: PyG batch object.
            task_type (Task): The current task type.

        Returns:
            Loss tensor.
        """
        # FIXME misha it is not task type, change to getting dvc field task
        if task_type == Task.GRAPH_CLASSIFICATION:
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, batch.batch)
            min_distances = self.gnn.min_distances

            # cluster loss
            self.prot_layer.prototype_class_identity = self.prot_layer.prototype_class_identity
            prototypes_of_correct_class = torch.t(
                self.prot_layer.prototype_class_identity[:, batch.y].bool())

            cluster_cost = torch.mean(
                torch.min(min_distances[prototypes_of_correct_class]
                          .reshape(-1, self.prot_layer.num_prototypes_per_class), dim=1)[0])

            # seperation loss
            separation_cost = -torch.mean(
                torch.min(min_distances[~prototypes_of_correct_class].reshape(-1, (
                        self.prot_layer.output_dim - 1) * self.prot_layer.num_prototypes_per_class),
                          dim=1)[0])

            # sparsity loss
            l1_mask = 1 - torch.t(self.prot_layer.prototype_class_identity)
            l1 = (self.prot_layer.last_layer.weight * l1_mask).norm(p=1)

            # diversity loss
            ld = 0
            # TODO expreriments required. With zero coeff - meaningless
            # for k in range(prot_layer.output_dim):
            #     p = prot_layer.prototype_vectors[
            #         k * prot_layer.num_prototypes_per_class:
            #         (k + 1) * prot_layer.num_prototypes_per_class]
            #     p = F.normalize(p, p=2, dim=1)
            #     matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]) - 0.3
            #     matrix2 = torch.zeros(matrix1.shape)
            #     ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            loss = self.loss_function(*move_to_same_device(logits, batch.y))
            loss += self.clst * cluster_cost + self.sep * separation_cost + 5e-4 * l1 + 0.00 * ld
            if self.clip is not None:
                clip_grad_norm(self.gnn.parameters(), self.clip)
            self.optimizer.zero_grad()
        elif task_type == Task.NODE_CLASSIFICATION:
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, batch.batch)
            loss = self.loss_function(*move_to_same_device(logits[:batch.batch_size], batch.y[:batch.batch_size]))

        elif task_type == Task.EDGE_PREDICTION:
            # TODO Kirill

            self.optimizer.zero_grad()
            edge_index = batch.edge_index
            pos_edge_index = edge_index[:, batch.y == 1]
            neg_edge_index = edge_index[:, batch.y == 0]

            pos_out = self.gnn(batch.x, pos_edge_index)
            neg_out = self.gnn(batch.x, neg_edge_index)

            # TODO check if we need to take out[:batch.batch_size]
            pos_loss = self.loss_function(*move_to_same_device(pos_out, torch.ones_like(pos_out)))
            neg_loss = self.loss_function(*move_to_same_device(neg_out, torch.zeros_like(neg_out)))

            loss = pos_loss + neg_loss
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        return loss

    def optimizer_step(
            self,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """ Backpropagate loss, clip gradients by value, and step the optimizer.
        """
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.gnn.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss

    def before_epoch(
            self,
            gen_dataset: GeneralDataset
    ):
        """
        Run prototype projection and configure gradient flags before each epoch.

        Args:
            gen_dataset (GeneralDataset): The training dataset.
        """
        cur_step = self.modification.epochs + 1
        train_ind = [n for n, x in enumerate(gen_dataset.train_mask) if x]
        # Prototype projection
        if cur_step > self.proj_epochs and cur_step % self.proj_epochs == 0:
            self.prot_layer.projection(self.gnn, gen_dataset.dataset, train_ind, gen_dataset.data,
                                       thrsh=self.prot_thrsh)
        self.gnn.train()
        for p in self.gnn.parameters():
            p.requires_grad = True
        self.prot_layer.prototype_vectors.requires_grad = True
        if cur_step < self.warm_epoch:
            for p in self.prot_layer.last_layer.parameters():
                p.requires_grad = False
        else:
            for p in self.prot_layer.last_layer.parameters():
                p.requires_grad = True

    def after_epoch(
            self,
            gen_dataset: GeneralDataset,
            **hook_kwargs
    ):
        """
        Evaluate validation metrics and update the best prototype cache after each epoch.

        Args:
            gen_dataset (GeneralDataset): Dataset used for evaluation.
            **hook_kwargs: Additional kwargs forwarded to the base class hook.
        """
        # TODO compare is_best with different metrics to be implemented

        # check if best model
        metrics_values = self.evaluate_model(
            gen_dataset, metrics=[
                Metric("Accuracy", mask='val'),
                Metric("Precision", mask='val'),
                Metric("Recall", mask='val')
            ]
        )
        self.cur_acc = metrics_values['val']["Accuracy"]
        self.is_best = (self.cur_acc - self.best_acc >= 0.01)

        if self.is_best:
            self.best_acc = self.cur_acc
            self.early_stop_count = 0
            self.gnn.best_prots = self.prot_layer.prototype_graphs

        super().after_epoch(gen_dataset, **hook_kwargs)

    def early_stopping(
            self,
            train_loss,
            gen_dataset: GeneralDataset,
            metrics: Union[List[Metric], Metric],
            steps: int
    ) -> bool:
        """
        Determine whether training should stop early.

        Stops if the model has not improved for early_stopping_marker consecutive epochs,
        or if the last prototype projection epoch has been reached.

        Args:
            train_loss: Training loss (required by base class interface, unused here).
            gen_dataset (GeneralDataset): Training dataset (unused).
            metrics (Union[List[Metric], Metric]): Metrics (unused).
            steps (int): Total number of training steps.

        Returns:
            True if training should stop, False otherwise.
        """
        step = self.modification.epochs
        if self.is_best:
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        last_projection = (step % self.proj_epochs == 0 and step + self.proj_epochs >= steps)

        return self.early_stop_count >= self.early_stopping_marker or last_projection
