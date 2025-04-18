import numpy as np
import shap
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from collections import OrderedDict

class ConceptEntropyMasker(torch.nn.Module):
    def __init__(self, explainer, epochs: int = 100, lr: float = 0.01, alpha: float = 0.001, beta: float = 0.1):
        super().__init__()
        self.explainer = explainer
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta

        self.concept_mask = None
        self.device = self.device

    def init_mask(self):
        self.concept_mask = torch.nn.Parameter(
            torch.randn(size=[self.model.n_units], requires_grad=True, device=self.device) * 0.1)

    def __loss__(self, logits, target):
        loss1 = cross_entropy(logits, target.long())

        M = self.concept_mask.sigmoid()
        loss2 = M.sum()  # regularisation

        ent = -M * torch.log(M + 1e-15) - (1 - M) * torch.log(1 - M + 1e-15)
        loss3 = ent.mean()

        # print(f'Loss1 {loss1 : .3f} Loss2: {loss2 : .3f} Loss3: {loss3 : .3f}')
        return loss1 + self.alpha * loss2 + self.beta * loss3

    def optimise(self, x, edge_index, target, show_progress=False):
        optimiser = torch.optim.Adam(params=[self.concept_mask], lr=self.lr)
        pbar = tqdm(range(self.epochs)) if show_progress else range(self.epochs)
        for idx in pbar:
            # time.sleep(0.2)
            concs = self.explainer.concept_layer(x, edge_index)

            top_level = self.model.lin1
            logits = top_level(concs * self.concept_mask.sigmoid())
            #logits =
            loss = self.__loss__(logits, target)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if show_progress:
                pbar.set_postfix_str(f'Epoch: {idx}   Loss: {loss : .3f}')

    def forward(self, x, edge_index, label, show_progress):
        self.init_mask()
        self.optimise(x.to(self.device), edge_index.to(self.device), target=torch.tensor([label]).to(self.device),
                      show_progress=show_progress)
        return self.concept_mask.detach()


def by_weight(explainer, label, layer=None):
    if layer is None:
        layer = explainer.model.model_info['last_graph_layer_ind'] - 1
    if layer != explainer.model.model_info['last_graph_layer_ind'] - 1:
        raise NotImplementedError
    else:
        # importance = model.lin1.weight[label].detach()
        weights = explainer.model.get_weights()
        weights = OrderedDict(explainer.model.get_weights())
        importance = torch.tensor(weights[next(reversed(weights))]['weight'][label]).detach()
        idxs = importance.argsort().flip([0])
        return idxs.cpu().numpy(), importance[idxs].cpu().numpy()


def by_weight_adv(model, label):
    importance = model.lin1.weight[label].detach() - model.lin1.weight[1 - label].detach()
    idxs = importance.argsort().flip([0])
    return idxs.cpu().numpy(), importance[idxs].cpu().numpy()


def by_weight_x_val(model, x, edge_index, label, sort=False):
    concept_layer = model.concept_layer(x.to(model.device), edge_index.to(model.device))[0]
    importance = (model.lin1.weight[label] * concept_layer).detach()

    if sort:
        idxs = importance.argsort().flip([0])
        return idxs.cpu().numpy(), importance[idxs].cpu().numpy()

    return importance.cpu().numpy()


def by_shap(model, train_dataset, x, edge_index, label):
    X_train = []

    for data in train_dataset:
        data = data.to(model.device)
        neurons = model.concept_layer(data.x, data.edge_index, data.batch)
        X_train.append(neurons.detach())

    X_test = model.concept_layer(x.to(model.device), edge_index.to(model.device))

    X_train = torch.row_stack(X_train)

    top_level = torch.nn.Sequential(
        model.lin1,
        torch.nn.Softmax()
    )

    explainer = shap.DeepExplainer(top_level, X_train)
    shap_values = explainer.shap_values(X_test)

    idxs = shap_values[label].squeeze().argsort()[::-1]
    scores = shap_values[label].squeeze()[idxs]

    return idxs, scores


def by_entropy(model, x, edge_index, label, epochs=100, lr=0.1, alpha=0.001, beta=0.1, debug=False, sort=False):
    cem = ConceptEntropyMasker(model, epochs=epochs, lr=lr, alpha=alpha, beta=beta)
    mask = cem(x=x, edge_index=edge_index, label=label, show_progress=debug)

    if sort:
        idxs = mask.argsort().flip([0])
        return idxs.cpu().numpy(), mask[idxs].sigmoid().cpu().numpy()

    return mask.sigmoid().cpu().numpy()


def by_entropy_x_val(model, x, edge_index, label, epochs=100, lr=0.1, alpha=0.001, beta=0.1, debug=False, sort=False):
    _, mask = by_entropy(model, x, edge_index, label, epochs=100, lr=0.1, alpha=0.001, beta=0.1, debug=False)
    concept_layer = model.concept_layer(x.to(model.device), edge_index.to(model.device))[0]
    mask = (torch.tensor(mask).to(model.device) * concept_layer).detach()
    idxs = mask.argsort().flip([0])
    return idxs.cpu().numpy(), mask[idxs].cpu().numpy()
