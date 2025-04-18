import numpy as np
import torch
import tqdm

import concept_ranker


def neuron_accuracy(model, data):
    model.eval()

    correct = 0
    results = []
    neuron_on = []

    for d in data:
        d.to(model.device)

        acts = model.partial_forward(d.x, d.edge_index, d.batch)
        neuron_on.append(acts.sum(dim=0) != 0)

        out = model.forward(d.x, d.edge_index, d.batch)
        pred = out.argmax(dim=1) if len(out.shape) > 1 else torch.round(out)
        correct += int((pred == d.y).sum())
        v = pred == d.y
        results.append(v)

        del d

    results = torch.cat(results)  # bool [n_graphs]
    neuron_on = torch.row_stack(neuron_on)  # bool mat [n_graphs, n_neurons]
    correct_conditioned = neuron_on & results[:, None]  # col broadcast
    neuron_accs = correct_conditioned.sum(dim=0) / (neuron_on.sum(dim=0) + 0.001)

    return neuron_accs


def neuron_x(model, data):
    model.eval()

    h_ = torch.zeros(model.n_units).to(model.device)
    count = 0
    for d in data:
        conc_layer = model.concept_layer(d.x.to(model.device), d.edge_index.to(model.device), d.batch)
        h = conc_layer * abs((model.lin1.weight[0].detach() + model.lin1.weight[0].detach()))
        h_ += h.sum(dim=0)
        count += h.shape[0]

    return (h_ / count).detach().cpu().numpy()


def neuron_y(model, data):
    model.eval()

    h_ = torch.zeros(model.n_units).to(model.device)
    count = 0
    for d in data:
        conc_layer = model.concept_layer(d.x.to(model.device), d.edge_index.to(model.device), d.batch)
        a = conc_layer * model.lin1.weight[0]
        b = conc_layer * model.lin1.weight[1]
        c = torch.cat((a.unsqueeze(0), b.unsqueeze(0))).detach()
        h = torch.std(c, dim=0)
        h_ += h.sum(dim=0)
        count += h.shape[0]

    return (h_ / count).detach().cpu().numpy()


def neuron_e(model, dataset):
    model.eval()

    h_ = torch.zeros(model.n_units).to(model.device)
    count = 0
    for g in tqdm(dataset):
        h_ += concept_ranker.by_entropy(model, g.x, g.edge_index, 0)
        count += 1
    return (h_ / count).detach().cpu().numpy()


def erasure(model, data_loader):
    model.eval()

    original_acc = model.evaluate(data_loader)

    drops = []

    for i in range(model.n_units):
        model.reset_erase()
        model.set_erase(model.n_layers - 1, i)

        a = model.evaluate(data_loader)
        drops.append(original_acc - a)

    return drops


def neuron_activity(model, data):
    model.eval()
    n_neurons = -1

    from collections import defaultdict
    counts = defaultdict(int)

    for d in data:
        d.to(model.device)

        acts = model.partial_forward(d.x, d.edge_index, d.batch)
        n_neurons = max(n_neurons, acts.shape[1])

        for k in range(n_neurons):
            if acts[:, k].sum() > 0:
                counts[k] += 1

    counts_ = {}
    for k, c in counts.items():
        counts_[k] = counts[k] / len(data)

    return counts_


def neuron_correlation(model, data):
    model.eval()

    losses = []
    neuron_on = []

    for d in data:
        d.to(model.device)

        acts = model.partial_forward(d.x, d.edge_index, d.batch)
        neuron_on.append(acts.mean(dim=0))

        out = model.forward(d.x, d.edge_index, d.batch).abs()
        v = - torch.nn.CrossEntropyLoss()(out, d.y.long()).detach()
        losses.append(v.unsqueeze(0))

        del d

    losses = torch.cat(losses)  # bool [n_graphs]
    neuron_on = torch.row_stack(neuron_on).T  # bool mat [n_neurons, n_graphs]
    M = torch.row_stack((neuron_on, losses))

    C = np.corrcoef(M.detach().cpu().numpy())
    C = np.nan_to_num(C)

    return C[:-1, -1]
