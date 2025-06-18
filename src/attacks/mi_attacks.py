import copy

import torch

from typing import Type, Union, List

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from attacks.attack_base import Attacker
from base.datasets_processing import GeneralDataset
from data_structures.configs import ModelConfig
from data_structures.mi_results import MIResultsStore
from models_builder.models_zoo import model_configs_zoo


class MIAttacker(
    Attacker
):
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
    ) -> float:
        """
        Computes accuracy for a single attack result (mask + inferred labels pair).

        Args:
            mask: Key from results store (boolean tensor form)
            inferred_labels: Value from results store (predicted labels tensor)
            mask_true: Tensor of true labels for all nodes in the graph

        Returns:
            float: Accuracy (0.0 to 1.0) of correct predictions among attacked samples
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
    name = "EmptyMIAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass


class NaiveMIAttacker(MIAttacker):
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

        model.eval()

        data = gen_dataset.data
        with torch.no_grad():
            probs = model.get_predictions(data.x, data.edge_index)
            max_probs = torch.max(probs, dim=1).values
            inferred_train_mask = (max_probs > self.thrsh) & mask_tensor
            self.results.add(mask_tensor, inferred_train_mask)

        return self.results


class ShadowModelMIAttacker(MIAttacker):
    name = "ShadowModelMIAttacker"

    def __init__(
            self,
            shadow_data_ratio: float = 0.25,  # Fraction of data to use for shadow training
            shadow_epochs: int = 100,  # Number of epochs to train shadow model
            classifier_type: str = 'svc',  # Type of classifier to use ('svc' or 'mlp')
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shadow_data_ratio = shadow_data_ratio
        self.shadow_epochs = shadow_epochs
        self.classifier_type = classifier_type
        self.classifier = None

        # TODO customizable surrogate model
        # TODO customizable classifier

    def _train_shadow_model(
            self,
            gen_dataset: GeneralDataset,
            shadow_train_mask: torch.Tensor
    ):
        """
        Train the shadow model on the shadow dataset
        """
        shadow_model = model_configs_zoo(dataset=gen_dataset, model_name=self.model_name)

        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.shadow_epochs):
            shadow_model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = shadow_model(gen_dataset.dataset.data.x, gen_dataset.dataset.data.edge_index)

            # Compute loss only on shadow training nodes
            loss = criterion(outputs[shadow_train_mask], gen_dataset.dataset.data.y[shadow_train_mask])
            print(f"Shadow loss: {loss}")

            # Backward pass
            loss.backward()
            optimizer.step()
        return shadow_model

    def _train_attack_classifier(
            self,
            shadow_model: torch.nn.Module,
            shadow_data: GeneralDataset,
            shadow_train_mask: torch.tensor
    ):
        """
        Train the attack classifier using shadow model outputs
        """
        shadow_model.eval()
        with torch.no_grad():
            outputs = shadow_model(shadow_data.dataset.data.x, shadow_data.dataset.data.edge_index)
            probs = torch.softmax(outputs, dim=1)
            # max_probs = torch.max(probs, dim=1).values.cpu().numpy()
        # Prepare features and labels for attack classifier
        # X = max_probs.reshape(-1, 1)  # Using prediction confidence as feature
        X = probs[shadow_train_mask].cpu().numpy()
        y = shadow_data.train_mask[shadow_train_mask].cpu().numpy().astype(int)  # Membership labels

        if self.classifier_type == 'svc':
            self.classifier = SVC(kernel='rbf', probability=False)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

        self.classifier.fit(X, y)

        # Evaluate on shadow data (for debugging)
        y_pred = self.classifier.predict(X)
        shadow_accuracy = accuracy_score(y, y_pred)
        print(f"Shadow model attack classifier accuracy: {shadow_accuracy:.4f}")

    def attack(
            self,
            model: torch.nn.Module,
            gen_dataset: GeneralDataset,
            mask_tensor: Union[List[bool], torch.Tensor],
            **kwargs
    ):
        task_type = gen_dataset.is_multi()
        if task_type:
            self.model_name = 'gcn_gcn_linear'
        else:
            self.model_name = 'gcn_gcn'

        dataset = copy.deepcopy(gen_dataset)

        num_nodes = dataset.dataset.data.x.shape[0]
        shadow_indices = torch.randperm(num_nodes)[:int(num_nodes * self.shadow_data_ratio)]
        shadow_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shadow_train_mask[shadow_indices[:int(len(shadow_indices) * 0.75)]] = True  # 75-25 split
        dataset.train_mask = shadow_train_mask
        dataset.test_mask = ~dataset.train_mask

        print("Training shadow model...")
        shadow_model = self._train_shadow_model(dataset, shadow_train_mask)

        print("Training attack classifier...")
        self._train_attack_classifier(shadow_model, dataset, shadow_train_mask)

        print("Performing attack on target model...")
        model.eval()
        with torch.no_grad():
            outputs = shadow_model(gen_dataset.dataset.data.x, gen_dataset.dataset.data.edge_index)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1).values.cpu().numpy()
        # Predict membership using attack classifier
        X_target = max_probs.reshape(-1, 1)
        inferred_train_mask = torch.tensor(self.classifier.predict(X_target),
                                           dtype=torch.bool, device=mask_tensor.device)

        # Store results
        self.results.add(mask_tensor, inferred_train_mask)

        return self.results