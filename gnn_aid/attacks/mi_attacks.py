import copy
from typing import Union, List, Tuple, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from gnn_aid.attacks.attack_base import Attacker
from gnn_aid.aux.utils import move_to_same_device
from gnn_aid.data_structures.mi_results import MIResultsStore
from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.models_builder import FrameworkGNNConstructor
from gnn_aid.models_builder.models_zoo import model_configs_zoo


class MIAttacker(
    Attacker
):
    """ Base class for all membership inference (MI) attack methods.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.results = MIResultsStore()

    @staticmethod
    def compute_single_attack_accuracy(
            mask: torch.Tensor,
            inferred_labels: torch.Tensor,
            mask_true: torch.Tensor,
            train_class_label: bool = True
    ) -> Union[float, Dict]:
        """
        Computes accuracy for a single attack result (mask + inferred labels pair).

        Args:
            mask: Key from results store (boolean tensor form)
            inferred_labels: Value from results store (predicted labels tensor)
            mask_true: Tensor of true labels for all nodes in the graph

        Returns:
            float: Dict with metrics for predictions among attacked samples.
                   Returns 0.0 if no samples were attacked
        """
        metrics = {
            'accuracy': 0.0,
            'precision_train': 0.0,
            'recall_train': 0.0,
            'f1_train': 0.0
        }

        attacked_indices = mask.nonzero().squeeze()

        if attacked_indices.numel() == 0:
            return 0.0

        true_labels = mask_true[attacked_indices]
        pred_labels = inferred_labels[attacked_indices]

        true_labels, pred_labels = move_to_same_device(true_labels, pred_labels)

        # Calculate overall accuracy
        correct = (true_labels == pred_labels).sum().item()
        metrics['accuracy'] = correct / len(attacked_indices)

        # Calculate train class metrics
        true_pos = ((pred_labels == train_class_label) & (true_labels == train_class_label)).sum().item()
        pred_pos = (pred_labels == train_class_label).sum().item()
        actual_pos = (true_labels == train_class_label).sum().item()

        # Precision
        metrics['precision_train'] = true_pos / pred_pos if pred_pos > 0 else 0.0

        # Recall
        metrics['recall_train'] = true_pos / actual_pos if actual_pos > 0 else 0.0

        # F1-score
        precision = metrics['precision_train']
        recall = metrics['recall_train']
        if (precision + recall) > 0:
            metrics['f1_train'] = 2 * (precision * recall) / (precision + recall)

        return metrics


class EmptyMIAttacker(
    MIAttacker
):
    """ Just a stub for MI attack.
    """
    name = "EmptyMIAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass


class NaiveMIAttacker(
    MIAttacker
):
    """
    Naive MI Attack: marks as train the data on which the model is most confident
    """
    name = "NaiveMIAttacker"

    def __init__(self, threshold: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.thrsh = threshold

    def attack(
        self,
        model: torch.nn.Module,  # more accurate typing?
        gen_dataset: GeneralDataset,
        mask_tensor: Union[str, List[bool], torch.Tensor],
    ):
        assert not isinstance(mask_tensor, str), "Input of original mask seems senseless"
        if isinstance(mask_tensor, list):
            mask_tensor = torch.tensor(mask_tensor)

        model.eval()

        data = gen_dataset.data
        with torch.no_grad():
            probs = model.get_predictions(data.x, data.edge_index)
            max_probs = torch.max(probs, dim=1).values
            inferred_train_mask = (max_probs > self.thrsh) & mask_tensor
            self.results.add(mask_tensor, inferred_train_mask)

        return self.results


class ShadowModelMIAttacker(
    MIAttacker
):
    """
    The surrogate model is trained on a part of the initial dataset.
    The classifier learns from its responses to determine whether the input is from train or test
    """
    name = "ShadowModelMIAttacker"

    def __init__(
            self,
            shadow_data_ratio: float = 0.25,
            shadow_train_ratio: float = 0.75,
            shadow_epochs: int = 100,
            classifier_type: str = 'svc',  # 'svc' only for now
            use_logits: bool = True,  # Use logits (recommended) or softmax probs
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shadow_data_ratio = shadow_data_ratio
        self.shadow_train_ratio = shadow_train_ratio
        self.shadow_epochs = shadow_epochs
        self.classifier_type = classifier_type
        self.use_logits = use_logits
        self.classifier = None
        self.model_name = None

    def _prepare_shadow_masks(
            self,
            num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create shadow train/test masks over a random subset of nodes.
        """
        all_indices = torch.arange(num_nodes)
        shadow_size = int(num_nodes * self.shadow_data_ratio)
        shadow_indices = all_indices[torch.randperm(num_nodes)[:shadow_size]]

        n_train = int(shadow_size * self.shadow_train_ratio)
        shadow_train_indices = shadow_indices[:n_train]
        shadow_test_indices = shadow_indices[n_train:]

        shadow_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shadow_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shadow_train_mask[shadow_train_indices] = True
        shadow_test_mask[shadow_test_indices] = True

        return shadow_train_mask, shadow_test_mask

    def _train_shadow_model(
            self,
            shadow_dataset: GeneralDataset,
            shadow_train_mask: torch.Tensor
    ) -> torch.nn.Module:
        """
        Train shadow model on shadow training nodes
        """
        shadow_model = model_configs_zoo(dataset=shadow_dataset, model_name=self.model_name)
        shadow_model = shadow_model.to(shadow_dataset.data.x.device)

        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.shadow_epochs):
            shadow_model.train()
            optimizer.zero_grad()

            outputs = shadow_model(shadow_dataset.data.x, shadow_dataset.data.edge_index)
            loss = criterion(
                *move_to_same_device(
                    outputs[shadow_train_mask],
                    shadow_dataset.data.y[shadow_train_mask]
                )
            )
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Shadow epoch {epoch}/{self.shadow_epochs}, loss: {loss.item():.4f}")

        return shadow_model

    def _extract_features(
            self,
            model: torch.nn.Module,
            dataset: GeneralDataset,
            node_mask: torch.Tensor
    ) -> np.ndarray:
        """Extract features (logits or probabilities) from model outputs."""
        model.eval()
        with torch.no_grad():
            outputs = model(dataset.data.x, dataset.data.edge_index)
            if self.use_logits:
                features = outputs[node_mask]
            else:
                features = torch.softmax(outputs[node_mask], dim=1)
            return features.cpu().numpy()

    def _train_attack_classifier(
            self,
            shadow_model: torch.nn.Module,
            shadow_dataset: GeneralDataset,
            shadow_train_mask: torch.Tensor,
            shadow_test_mask: torch.Tensor
    ):
        """Train attack classifier using shadow model outputs."""
        X_train = self._extract_features(shadow_model, shadow_dataset, shadow_train_mask)
        y_train = np.ones(X_train.shape[0])

        X_test = self._extract_features(shadow_model, shadow_dataset, shadow_test_mask)
        y_test = np.zeros(X_test.shape[0])

        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        # Train classifier
        if self.classifier_type == 'svc':
            self.classifier = SVC(kernel='rbf', probability=True)
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        self.classifier.fit(X, y)

    def attack(
            self,
            model: torch.nn.Module,
            gen_dataset: GeneralDataset,
            mask_tensor: Union[List[bool], torch.Tensor],
            **kwargs
    ):
        """
        Perform membership inference attack using shadow model technique.
        """
        if isinstance(mask_tensor, list):
            mask_tensor = torch.tensor(mask_tensor)
        task_type = gen_dataset.is_multi()
        self.model_name = 'gcn_gcn_lin_no_softmax' if task_type else 'gcn_gcn_no_softmax'

        num_nodes = gen_dataset.data.x.shape[0]
        shadow_indices = torch.randperm(num_nodes)[:int(num_nodes * self.shadow_data_ratio)]
        n_shadow = len(shadow_indices)
        n_train = int(n_shadow * 0.75)

        shadow_train_indices = shadow_indices[:n_train]
        shadow_test_indices = shadow_indices[n_train:]

        shadow_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shadow_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shadow_train_mask[shadow_train_indices] = True
        shadow_test_mask[shadow_test_indices] = True

        shadow_dataset = copy.deepcopy(gen_dataset)
        shadow_dataset.train_mask = shadow_train_mask
        shadow_dataset.test_mask = shadow_test_mask

        shadow_model = self._train_shadow_model(shadow_dataset, shadow_train_mask)

        self._train_attack_classifier(shadow_model, shadow_dataset, shadow_train_mask, shadow_test_mask)

        model.eval()
        with torch.no_grad():
            outputs = model(gen_dataset.data.x, gen_dataset.data.edge_index)
            if self.use_logits:
                features = outputs
            else:
                features = torch.softmax(outputs, dim=1)
            features = features.cpu().numpy()

        all_predictions = self.classifier.predict(features)
        inferred_membership_full = torch.tensor(all_predictions, dtype=torch.bool)

        self.results.add(mask_tensor, inferred_membership_full)
        return self.results


class ShadowModelMILinkAttacker(MIAttacker):
    """
    Shadow model-based membership inference attack for Link Prediction.
    """
    name = "ShadowModelMILinkAttacker"

    def __init__(
            self,
            shadow_edge_ratio: float = 0.2,
            shadow_train_ratio: float = 0.75,
            shadow_epochs: int = 10,
            classifier_type: str = 'linreg',
            use_embedding_features: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shadow_edge_ratio = shadow_edge_ratio
        self.shadow_train_ratio = shadow_train_ratio
        self.shadow_epochs = shadow_epochs
        self.classifier_type = classifier_type
        self.use_embedding_features = use_embedding_features
        self.classifier = None
        self.model_name = None

    def _prepare_shadow_edge_masks(
            self,
            num_edges: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create shadow train/test masks over a random subset of edges
        """
        all_indices = torch.arange(num_edges)
        shadow_size = int(num_edges * self.shadow_edge_ratio)
        shadow_indices = all_indices[torch.randperm(num_edges)[:shadow_size]]

        n_train = int(shadow_size * self.shadow_train_ratio)
        shadow_train_indices = shadow_indices[:n_train]
        shadow_test_indices = shadow_indices[n_train:]

        shadow_train_mask = torch.zeros(num_edges, dtype=torch.bool)
        shadow_test_mask = torch.zeros(num_edges, dtype=torch.bool)
        shadow_train_mask[shadow_train_indices] = True
        shadow_test_mask[shadow_test_indices] = True

        return shadow_train_mask, shadow_test_mask

    def _train_shadow_model(
            self,
            shadow_model: torch.nn.Module,
            shadow_dataset: GeneralDataset,
            shadow_train_mask: torch.Tensor,
            device: torch.device
    ) -> torch.nn.Module:
        """
        Train shadow model on shadow dataset
        """
        shadow_model = shadow_model.to(device)
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.shadow_epochs):
            shadow_model.train()
            optimizer.zero_grad()

            node_emb = shadow_model(
                shadow_dataset.data.x.to(device),
                shadow_dataset.data.edge_index.to(device)
            )

            train_edge_index = shadow_dataset.edge_label_index[:, shadow_train_mask].to(device)
            train_edge_labels = shadow_dataset.edge_labels[shadow_train_mask].float().to(device)

            edge_logits = shadow_model.decode(node_emb[train_edge_index[0]], node_emb[train_edge_index[1]]).squeeze()

            loss = criterion(edge_logits, train_edge_labels)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Shadow epoch {epoch}/{self.shadow_epochs}, loss: {loss.item():.4f}")

        return shadow_model

    def _extract_edge_features(
            self,
            model: torch.nn.Module,
            dataset: GeneralDataset,
            edge_mask: torch.Tensor,
            device: torch.device
    ) -> np.ndarray:
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            node_emb = model(
                dataset.data.x.to(device),
                dataset.data.edge_index.to(device)
            )

            edge_index = dataset.edge_label_index[:, edge_mask].to(device)

            edge_logits = model.decode(node_emb[edge_index[0]], node_emb[edge_index[1]]).squeeze()
            edge_probs = torch.sigmoid(edge_logits)

            if self.use_embedding_features:
                features = torch.cat([
                    edge_probs.unsqueeze(1),
                    node_emb[edge_index[0]],
                    node_emb[edge_index[1]]
                ], dim=1)
            else:
                features = edge_probs.unsqueeze(1)

            return features.cpu().numpy()

    def _train_attack_classifier(
            self,
            shadow_model: torch.nn.Module,
            shadow_dataset: GeneralDataset,
            shadow_train_mask: torch.Tensor,
            shadow_test_mask: torch.Tensor,
            device: torch.device
    ):
        """
        Train attack classifier using shadow model outputs
        """
        X_train = self._extract_edge_features(shadow_model, shadow_dataset, shadow_train_mask, device)
        y_train = np.ones(X_train.shape[0])

        X_test = self._extract_edge_features(shadow_model, shadow_dataset, shadow_test_mask, device)
        y_test = np.zeros(X_test.shape[0])

        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        if self.classifier_type == 'svc':
            self.classifier = SVC(kernel='rbf', probability=True)
        elif self.classifier_type == 'linreg':
            self.classifier = LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        self.classifier.fit(X, y)

    def attack(
            self,
            model: torch.nn.Module,
            gen_dataset: GeneralDataset,
            mask_tensor: Union[torch.Tensor, list],
            **kwargs
    ):
        """
        Perform membership inference attack on target model
        """
        if isinstance(mask_tensor, str):
            if mask_tensor == 'train':
                mask_tensor = gen_dataset.train_mask
            elif mask_tensor == 'val':
                mask_tensor = gen_dataset.val_mask
            elif mask_tensor == 'test':
                mask_tensor = gen_dataset.test_mask
            elif mask_tensor == 'all':
                mask_tensor = torch.ones(
                    gen_dataset.edge_label_index.size(1),
                    dtype=torch.bool,
                    device=gen_dataset.train_mask.device
                )
            else:
                raise ValueError(f"Unknown mask string: {mask_tensor}")

        num_edges = gen_dataset.edge_label_index.size(1)
        device = next(model.parameters()).device

        shadow_train_mask, shadow_test_mask = self._prepare_shadow_edge_masks(num_edges)

        shadow_dataset = copy.deepcopy(gen_dataset)
        shadow_dataset.train_mask = shadow_train_mask
        shadow_dataset.test_mask = shadow_test_mask

        shadow_model = model_configs_zoo(dataset=shadow_dataset, model_name='gcn_link_pred')
        shadow_model = self._train_shadow_model(shadow_model, shadow_dataset, shadow_train_mask, device)

        self._train_attack_classifier(
            shadow_model, shadow_dataset, shadow_train_mask, shadow_test_mask, device
        )

        target_features = self._extract_edge_features(model, gen_dataset, torch.ones(num_edges, dtype=torch.bool),
                                                      device)

        all_predictions = self.classifier.predict(target_features)
        inferred_membership_full = torch.tensor(all_predictions, dtype=torch.bool)

        self.results.add(mask_tensor, inferred_membership_full)
        members_count = inferred_membership_full.sum().item()
        return self.results
