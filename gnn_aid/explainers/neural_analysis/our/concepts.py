import numpy as np
import torch
import torch_scatter
from collections import defaultdict

from gnn_aid.explainers.neural_analysis.orig import concepts

class ConceptSet(concepts.ConceptSet):
    def __init__(self, graphs, task, omega=[10, 20, 20], pbar=None, device=None):
        if device is None:
            device = "cpu"
        self.device = device
        if task in ['MUTAG', 'BBBP', 'MUTAGENICITY', 'PROTEINS', 'NCI', 'BA', 'IMDB', 'REDDIT', 'SST']:
            super().__init__(graphs, task, omega)
        else:
            self.num_graphs = len(graphs)
            self.graphs = graphs
            self.base_concepts = defaultdict(lambda: [None, defaultdict(lambda: None)])
            self.neuron_concept_tracker = {}  # maps neuron to list of lists [name, score, concept]
            self.cur_length = 1  # current formula length
            self.omega = omega

            print('Constructing base concepts')
            if task == 'BA2Motif':
                concs = [
                    [concepts.BaseConcept(f'ndeg={i}', lambda g, i=i: concepts.concepts_basic.neigh_degree(g, k=i, operator='=', require=1)
                             ) for i in [1, 2, 3]],
                    [concepts.BaseConcept(f'deg-greater-than{i}', lambda g, i=i: concepts.concepts_basic.degree(g, k=i, operator='>'))
                     for i in range(1, 100, 2)]
                ]
            else:
                raise Exception('Unknown task %s' % task)

            for group_idx, group in enumerate(concs):
                if isinstance(group, concepts.BaseConcept):
                    group = [group]
                for concept in group:
                    concept.group = group_idx
                    self.base_concepts[concept.name()][0] = concept
                    for graph_idx, graph in enumerate(graphs):
                        concept_mask = concept(graph)
                        self.base_concepts[concept.name()][1][graph_idx] = concept_mask
                    self.compose(concept, concept, 'inv')

            self.clean_concepts()
        self.pbar = pbar

    def expand(self):
        marked_for_deletion = defaultdict(int)

        # only keep candidates within the beam by removing all others.
        if self.cur_length > 1:
            for neuron in self.neuron_concept_tracker.keys():
                tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                    c.length() == self.cur_length]
                idxs1 = np.argsort([v[0] for [v, _] in tracked_concepts])

                for i in idxs1[-concepts.BEAM_WIDTH:]:
                    marked_for_deletion[tracked_concepts[i][1].name()] += 1

            deleted_set = set()
            #self.pbar.reset(len(list(self.neuron_concept_tracker.keys())))
            for neuron in self.neuron_concept_tracker.keys():
                tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                    c.length() == self.cur_length]
                for _, c in tracked_concepts:
                    c_name = c.name()
                    if c.length() == self.cur_length and marked_for_deletion[c_name] == 0 and c_name not in deleted_set:
                        self.base_concepts[c_name][1].clear()
                        deleted_set.add(c_name)
                #self.pbar.update(1)

            print(f'Deleted: {len(deleted_set)}')

        pruned = 0

        # expand to the next depth
        #self.pbar.reset(len(list(self.neuron_concept_tracker.keys())))
        for neuron in self.neuron_concept_tracker.keys():
            tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                c.length() == self.cur_length]
            unary_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if c.length() == 1]

            idxs1 = np.argsort([v[0] for [v, _] in tracked_concepts])[-concepts.BEAM_WIDTH:]
            idxs2 = np.argsort([v[0] for [v, _] in unary_concepts])

            for i in idxs1:
                for j in idxs2:
                    concept_a = tracked_concepts[i][1]
                    concept_b = unary_concepts[j][1]

                    common_groups = concept_a.concept_groups().intersection(concept_b.concept_groups())
                    if len(common_groups) > 0:
                        continue

                    if concept_a.length() + concept_b.length() != self.cur_length + 1:
                        continue

                    for op in ['inter', 'union']:
                        new_conc = self.compose(concept_a, concept_b, op)
                        if self.check_clean(new_conc[1].name()):
                            del self.base_concepts[new_conc[1].name()]
                            pruned += 1
                        else:
                            self.neuron_concept_tracker[neuron].append(new_conc)
            self.pbar.update(1)

        self.cur_length += 1

        print(f'Pruned: {pruned} concepts!')

    def match(self, neuron_activations, norm, inds):
        """
        Updates score for each concept for all neurons.
        :param neuron_activations:  #neurons x #nodes, all graphs stacked, normalised
        :param norm: #nodes x 1, normalisation term
        """

        n_nodes = neuron_activations.shape[1]

        neuron_activations = neuron_activations.to(self.device)
        inds = inds.to(self.device)

        #self.pbar.reset(neuron_activations.shape[0])
        for neuron_idx in range(neuron_activations.shape[0]):
            if neuron_idx not in self.neuron_concept_tracker:
                self.neuron_concept_tracker[neuron_idx] = []
                for _, [concept, _] in self.base_concepts.items():
                    self.neuron_concept_tracker[neuron_idx].append([None, concept])

            unscored_concepts = self.get_unscored(neuron_idx)
            n_concepts = len(unscored_concepts)

            if n_concepts == 0:
                continue

            targets = self.truth(neuron_idx, unscored_concepts)
            activations = neuron_activations[neuron_idx, :]  # 1D, all activations for neuron

            n_graphs_with_acts = torch_scatter.scatter_add(activations, inds).nonzero().shape[0]
            max_act = torch_scatter.scatter_max(activations, inds)[0].sum()
            if n_graphs_with_acts > 0:
                max_act /= n_graphs_with_acts

            activations = activations.repeat(n_concepts, 1)

            assert activations.shape[0] == n_concepts and activations.shape[1] == n_nodes
            assert targets.shape[0] == n_concepts and targets.shape[1] == n_nodes

            err = torch.zeros(n_concepts).to(self.device)
            targets = targets.to(self.device)

            [alpha, beta, gamma] = self.omega

            final_thresholds = torch.zeros(activations.shape[0], device=self.device)

            for thresh in np.arange(alpha, beta) / gamma * max_act.item():
                acts = activations > thresh

                msk = torch.zeros(activations.shape, device=self.device)
                nonzids = acts.nonzero().T
                if nonzids.shape[0] > 0:
                    msk[nonzids[0], nonzids[1]] = 1
                act_th = activations * msk

                inters = torch_scatter.scatter_add(torch.logical_and(acts, targets).float(), inds)
                unions = torch_scatter.scatter_add(torch.logical_or(acts, targets).float(), inds)

                framed = torch_scatter.scatter_add(torch.logical_and(acts, targets).float() * act_th, inds)
                signal = torch_scatter.scatter_add(act_th, inds)

                frac = torch.nan_to_num((inters / unions) * (framed / signal), 0, 0, 0)

                denom = torch.logical_or(inters != 0, unions != 0).sum(dim=1)
                e_ = frac.sum(dim=1) / denom
                final_thresholds[torch.where(e_ > err)] = thresh
                err = torch.maximum(err, e_)

            assert err.shape[0] == n_concepts and len(err.shape) == 1

            for i, concept_idx in enumerate(unscored_concepts):
                self.neuron_concept_tracker[neuron_idx][concept_idx][0] = (err[i].item(), final_thresholds[i].item())

            del targets
            #self.pbar.update(1)

        del neuron_activations

        ret_dic = {}
        for neuron_idx in self.neuron_concept_tracker.keys():
            ret_dic[neuron_idx] = {}
            for [val, concept] in self.neuron_concept_tracker[neuron_idx]:
                ret_dic[neuron_idx][concept.name()] = (concept, val[0], val[1])
        return ret_dic
