from pathlib import Path
from typing import Type, Union

import torch
from torch.nn.utils import clip_grad_norm

from gnn_aid.aux.utils import OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH, \
    move_to_same_device
from gnn_aid.data_structures import Task
from gnn_aid.data_structures.configs import ConfigPattern
from . import FrameworkGNNModelManager


class GSATModelManager(FrameworkGNNModelManager):
    additional_config = ConfigPattern(  # TODO check config
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
            gnn: Type = None,
            dataset_path: Union[str, Path] = None,
            learn_edge_features: bool = False,
            decay_int: int = 10,
            decay_r: float = 0.1,
            init_r: float = 0.9,
            final_r: float = 0.1,
            fix_r: bool = False,
            pred_loss_coef: float = 1,
            info_loss_coef: float = 1,
            **kwargs
    ):
        super().__init__(gnn=gnn, dataset_path=dataset_path, **kwargs)
        self.learn_edge_features = learn_edge_features
        # TODO check if learn_edge_features == True then model -> GINEConv or other compatible
        self.decay_int = decay_int
        self.decay_r = decay_r
        self.init_r = init_r
        self.final_r = final_r
        self.pred_loss_coef = pred_loss_coef
        self.info_loss_coef = info_loss_coef
        self.fix_r = fix_r
        self.gsat_layer = getattr(self.gnn, self.gnn.gsat_layer_name)
        # apply_decorator_to_graph_layers(self.gnn, apply_attention)

    def train_on_batch(
            self,
            batch,
            task_type: Task
    ) -> torch.Tensor:
        if task_type == Task.GRAPH_CLASSIFICATION:
            self.optimizer.zero_grad()
            clf_logits = self.gnn(batch.x, batch.edge_index, batch.batch)
            att = self.gsat_layer.edge_att
            loss = self.gsat_loss(*move_to_same_device(att, clf_logits, batch.y, self.modification.epochs))
            # loss = self.loss_function(clf_logits, batch.y)
            self.optimizer.zero_grad()
            # del self.gsat_layer.att
        elif task_type == Task.NODE_CLASSIFICATION:
            self.optimizer.zero_grad()
            clf_logits = self.gnn(batch.x, batch.edge_index)  # TODO check weight param
            att = self.gsat_layer.edge_att
            # att = torch.zeros_like(clf_logits)
            # Take only predictions and labels of seed nodes
            loss = self.gsat_loss(*move_to_same_device(att, clf_logits[:batch.batch_size], batch.y[:batch.batch_size], self.modification.epochs))
            if self.clip is not None:
                clip_grad_norm(self.gnn.parameters(), self.clip)
            self.optimizer.zero_grad()
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
            raise ValueError(f"Unsupported task type: {task_type}")
        return loss

    def gsat_loss(
            self,
            att: torch.Tensor,
            logits: torch.Tensor,
            labels: torch.Tensor,
            epoch: int
    ) -> torch.Tensor:
        pred_loss = self.loss_function(logits, labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_int, self.decay_r, epoch, final_r=self.final_r,
                                                     init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        # self.info_loss_coef = 0
        info_loss = info_loss * self.info_loss_coef
        print(self.pred_loss_coef, self.info_loss_coef, pred_loss, info_loss)
        loss = pred_loss + info_loss
        return loss

    def get_r(
            self,
            decay_interval: float,
            decay_r: float,
            current_epoch: int,
            init_r=0.9,
            final_r=0.5
    ) -> float:
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
