import copy
import numpy as np
import torch
import torch_scatter
from collections import defaultdict

from gnn_aid.explainers.neural_analysis.orig.graph_utils import (
    edge_index_to_adj_list, edge_index_to_tuples)

BEAM_WIDTH = 10


# a set of base concepts
class ConceptSet:
    def __init__(self, graphs, task, omega=[10, 20, 20]):
        self.num_graphs = len(graphs)
        self.graphs = graphs
        self.base_concepts = defaultdict(lambda: [None, defaultdict(lambda: None)])
        self.neuron_concept_tracker = {}  # maps neuron to list of lists [name, score, concept]
        self.cur_length = 1  # current formula length
        self.omega = omega

        print('Constructing base concepts')

        if task == 'MUTAG':
            concs = [
                BaseConcept('isC ', lambda g: concepts_mutag.is_element(g, 'C')),
                BaseConcept('isN ', lambda g: concepts_mutag.is_element(g, 'N')),
                BaseConcept('isO ', lambda g: concepts_mutag.is_element(g, 'O')),
                BaseConcept('isF ', lambda g: concepts_mutag.is_element(g, 'F')),
                BaseConcept('isI ', lambda g: concepts_mutag.is_element(g, 'I')),
                BaseConcept('isCl', lambda g: concepts_mutag.is_element(g, 'Cl')),
                BaseConcept('isBr', lambda g: concepts_mutag.is_element(g, 'Br')),

                BaseConcept('nxC ', lambda g: concepts_mutag.element_neighbour(g, 'X', 'C', strict=False)),
                BaseConcept('nxN ', lambda g: concepts_mutag.element_neighbour(g, 'X', 'N', strict=False)),
                BaseConcept('nxF ', lambda g: concepts_mutag.element_neighbour(g, 'X', 'F', strict=False)),
                BaseConcept('nxO ', lambda g: concepts_mutag.element_neighbour(g, 'X', 'O', strict=False)),
                BaseConcept('nxCl', lambda g: concepts_mutag.element_neighbour(g, 'X', 'Cl', strict=False)),
                BaseConcept('nxBr', lambda g: concepts_mutag.element_neighbour(g, 'X', 'Br', strict=False)),
                BaseConcept('nxI ', lambda g: concepts_mutag.element_neighbour(g, 'X', 'I', strict=False)),

                [BaseConcept('nx1C', lambda g: concepts_mutag.element_neighbour(g, 'X', 'C', strict=True)),
                 BaseConcept('nx2C', lambda g: concepts_mutag.element_neighbour(g, 'X', 'C', 'C', strict=True)),
                 BaseConcept('nx3C', lambda g: concepts_mutag.element_neighbour(g, 'X', 'C', 'C', 'C', strict=True)
                             )],

                [BaseConcept(f'deg={i}', lambda g, i=i: concepts_basic.degree(g, k=i, operator='=')) for i in
                 range(1, 4)],
                [BaseConcept(f'ndeg={i}', lambda g, i=i: concepts_basic.neigh_degree(g, k=i, operator='=', require=1)
                             ) for i in [1, 2, 3]]
            ]
        elif task == 'BBBP':
            concs = [
                [BaseConcept(f'feature0-greater-than{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=0, k=k, operator='>')) for k in range(1, 6)],
                [BaseConcept(f'feature2-greater-than{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=2, k=k, operator='>')) for k in range(1, 6)],
                [BaseConcept(f'feature3-greater-than{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=3, k=k, operator='>')) for k in range(1, 6)],
                [BaseConcept(f'feature4-greater-than{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=4, k=k, operator='>')) for k in range(1, 6)],
                [BaseConcept(f'feature6-greater-than{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=6, k=k, operator='>')) for k in range(1, 6)],
                [BaseConcept(f'feature7-is-{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=7, k=k, operator='=')) for k in range(1, 2)],
                [BaseConcept(f'feature8-is-{k}',
                             lambda g, k=k: concepts_basic.feature(g, j=8, k=k, operator='=')) for k in range(1, 2)],
            ]
        elif task == 'MUTAGENICITY':
            concs = [
                BaseConcept('no2 ', lambda g: concepts_mutagenicity.AB_K(g, 'N', 'O', 2)),
                BaseConcept('nh2 ', lambda g: concepts_mutagenicity.AB_K(g, 'N', 'H', 2)),
                BaseConcept('no1 ', lambda g: concepts_mutagenicity.AB_K(g, 'N', 'O', 1)),

                BaseConcept('isC ', lambda g: concepts_mutagenicity.is_element(g, 'C')),
                BaseConcept('isN ', lambda g: concepts_mutagenicity.is_element(g, 'N')),
                BaseConcept('isO ', lambda g: concepts_mutagenicity.is_element(g, 'O')),
                BaseConcept('isF ', lambda g: concepts_mutagenicity.is_element(g, 'F')),
                BaseConcept('isI ', lambda g: concepts_mutagenicity.is_element(g, 'I')),
                BaseConcept('isCl', lambda g: concepts_mutagenicity.is_element(g, 'Cl')),
                BaseConcept('isS', lambda g: concepts_mutagenicity.is_element(g, 'S')),
                BaseConcept('isK', lambda g: concepts_mutagenicity.is_element(g, 'K')),
                BaseConcept('isLi', lambda g: concepts_mutagenicity.is_element(g, 'Li')),
                BaseConcept('isP', lambda g: concepts_mutagenicity.is_element(g, 'P')),
                BaseConcept('isH', lambda g: concepts_mutagenicity.is_element(g, 'H')),
                BaseConcept('isCa', lambda g: concepts_mutagenicity.is_element(g, 'Ca')),
                BaseConcept('isBr', lambda g: concepts_mutagenicity.is_element(g, 'Br')),

                BaseConcept('nxC ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'C', strict=False)),
                BaseConcept('nxN ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'N', strict=False)),
                BaseConcept('nxF ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'F', strict=False)),
                BaseConcept('nxO ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'O', strict=False)),
                BaseConcept('nxCl', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Cl', strict=False)
                            ),
                BaseConcept('nxBr', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Br', strict=False)
                            ),
                BaseConcept('nxI ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'I', strict=False)),
                BaseConcept('nxC ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'C', strict=False)),
                BaseConcept('nxS ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'S', strict=False)),
                BaseConcept('nxK ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'K', strict=False)),
                BaseConcept('nxLi', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Li', strict=False)
                            ),
                BaseConcept('nxP ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'P', strict=False)),
                BaseConcept('nxH ', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'H', strict=False)),
                BaseConcept('nxCa', lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Ca', strict=False)
                            ),

                BaseConcept('2hpC',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'C', strict=False, hops=2)),
                BaseConcept('2hpN',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'N', strict=False, hops=2)),
                BaseConcept('2hpF',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'F', strict=False, hops=2)),
                BaseConcept('2hpO',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'O', strict=False, hops=2)),
                BaseConcept('2hCl',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Cl', strict=False, hops=2)
                            ),
                BaseConcept('2hBr',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Br', strict=False, hops=2)
                            ),
                BaseConcept('2hpI',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'I', strict=False, hops=2)),
                BaseConcept('2hpC',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'C', strict=False, hops=2)),
                BaseConcept('2hpS',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'S', strict=False, hops=2)),
                BaseConcept('2hpK',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'K', strict=False, hops=2)),
                BaseConcept('2hLi',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Li', strict=False, hops=2)
                            ),
                BaseConcept('2hpP ',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'P', strict=False, hops=2)),
                BaseConcept('2hpH',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'H', strict=False, hops=2)),
                BaseConcept('2hCa',
                            lambda g: concepts_mutagenicity.element_neighbour(g, 'X', 'Ca', strict=False, hops=2)),
            ]
        elif task == 'PROTEINS':
            concs = [
                BaseConcept('isA_', lambda g: concepts_prot.is_element(g, 'A')),
                BaseConcept('isB_', lambda g: concepts_prot.is_element(g, 'B')),
                BaseConcept('isC_', lambda g: concepts_prot.is_element(g, 'C')),

                [BaseConcept('nx1A', lambda g: concepts_prot.element_neighbour(g, 'X', 'A', strict=False)),
                 BaseConcept('nx2A', lambda g: concepts_prot.element_neighbour(g, 'X', 'A', 'A', strict=False)),
                 BaseConcept('nx3A', lambda g: concepts_prot.element_neighbour(g, 'X', 'A', 'A', 'A', strict=False))],

                [BaseConcept('nx1B', lambda g: concepts_prot.element_neighbour(g, 'X', 'B', strict=False)),
                 BaseConcept('nx2B', lambda g: concepts_prot.element_neighbour(g, 'X', 'B', 'B', strict=False)),
                 BaseConcept('nx3B', lambda g: concepts_prot.element_neighbour(g, 'X', 'B', 'B', 'B', strict=False))],

                [BaseConcept('nx1C', lambda g: concepts_prot.element_neighbour(g, 'X', 'C', strict=False)),
                 BaseConcept('nx2C', lambda g: concepts_prot.element_neighbour(g, 'X', 'C', 'C', strict=False)),
                 BaseConcept('nx3C', lambda g: concepts_prot.element_neighbour(g, 'X', 'C', 'C', 'C', strict=False)),
                 BaseConcept('nx4C',
                             lambda g: concepts_prot.element_neighbour(g, 'X', 'C', 'C', 'C', 'C', strict=False)),
                 BaseConcept('nx5C',
                             lambda g: concepts_prot.element_neighbour(g, 'X', 'C', 'C', 'C', 'C', strict=False))],

                [BaseConcept('nb-nx-1A', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'A',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-2A', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'A',
                                                                                                                   'A',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-3A', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'A',
                                                                                                                   'A',
                                                                                                                   'A',
                                                                                                                   strict=False)))],

                [BaseConcept('nb-nx-1B', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'B',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-2B', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'B',
                                                                                                                   'B',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-3B', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'B',
                                                                                                                   'B',
                                                                                                                   'B',
                                                                                                                   strict=False)))],

                [BaseConcept('nb-nx-1C', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'C',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-2C', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'C',
                                                                                                                   'C',
                                                                                                                   strict=False))),
                 BaseConcept('nb-nx-3C', lambda g1: higher_order_wrapper(g1,
                                                                         lambda g: concepts_prot.element_neighbour(g,
                                                                                                                   'X',
                                                                                                                   'C',
                                                                                                                   'C',
                                                                                                                   'C',
                                                                                                                   strict=False)))]
            ]
        elif task == 'NCI':
            concs = [
                *[BaseConcept(f'is{i}', lambda g, i=i: concepts_nci.is_element(g, str(i))) for i in range(37)],
                *[BaseConcept(f'nx{i}', lambda g, i=i: concepts_nci.element_neighbour(g, 'X', str(i), strict=False)) for
                  i in range(37)],
                *[BaseConcept(f'2hp{i}',
                              lambda g, i=i: concepts_nci.element_neighbour(g, 'X', str(i), strict=False, hops=2))
                  for i in range(37)],
            ]
        elif task == 'BA':
            concs = [
                [BaseConcept(f'big{i}', lambda g, i=i: concepts_basic.degree(g, k=i, operator='>')) for i in
                 range(1, 100, 3)],
                [BaseConcept(f'ndeg{i}', lambda g, i=i: concepts_basic.neigh_degree(g, k=30, operator='>', require=i))
                 for i in range(1, 3)],
                [BaseConcept(f'NDEG{i}', lambda g, i=i: concepts_basic.neigh_degree(g, k=1, operator='=', require=i))
                 for i in [1, 2, 3, 4, 5, 10]]]
        elif task == 'IMDB':
            concs = [
                [BaseConcept(f'deg-greater-than{i}', lambda g, i=i: concepts_basic.degree(g, k=i, operator='>'))
                 for i in range(1, 100, 2)],
                [BaseConcept(f'has-nb-deg-greater-than-10{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=10, operator='>', require=i)) for i
                 in [1, 2, 3, 4, 5]],
                [BaseConcept(f'has-nb-deg-greater-than-30{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=30, operator='>', require=i)) for i
                 in [1, 2, 3, 4, 5]]]
        elif task == 'REDDIT':
            concs = [
                [BaseConcept(f'deg-greater-than{i}', lambda g, i=i: concepts_basic.degree(g, k=i, operator='>'))
                 for i in range(1, 100, 3)],
                [BaseConcept(f'has-nb-deg-greater-than-5{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=5, operator='>', require=i)) for i in
                 [1, 2]],
                [BaseConcept(f'has-nb-deg-greater-than-10{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=10, operator='>', require=i)) for i
                 in [1, 2]],
                [BaseConcept(f'has-nb-deg-greater-than-30{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=30, operator='>', require=i)) for i
                 in [1, 2]],
                [BaseConcept(f'has-nb-with-degree-one{i}',
                             lambda g, i=i: concepts_basic.neigh_degree(g, k=1, operator='=', require=i)) for i in
                 [1, 2, 3, 4, 5, 10]]]
        elif task == 'SST':
            concs = [
                [BaseConcept(f'big{i}', lambda g, i=i: concepts_basic.degree(g, k=i, operator='>')) for i in
                 range(1, 100, 3)],
                [BaseConcept(f'ndeg{i}', lambda g, i=i: concepts_basic.neigh_degree(g, k=30, operator='>', require=i)
                             ) for i in range(1, 10)],
                [BaseConcept(f'NDEG{i}', lambda g, i=i: concepts_basic.neigh_degree(g, k=1, operator='=', require=i)
                             ) for i in [1, 2, 3, 4, 5, 10]]]
        else:
            raise Exception('Unknown task %s' % task)

        for group_idx, group in enumerate(concs):
            if isinstance(group, BaseConcept):
                group = [group]
            for concept in group:
                concept.group = group_idx
                self.base_concepts[concept.name()][0] = concept
                for graph_idx, graph in enumerate(graphs):
                    concept_mask = concept(graph)
                    self.base_concepts[concept.name()][1][graph_idx] = concept_mask
                self.compose(concept, concept, 'inv')

        self.clean_concepts()

        # for name, [obj, _] in self.base_concepts.items():
        #     print(f'{name}  {obj.concept_groups()}')

    def random_concept_set(self, n_concepts, max_length=3):
        cs = [v[0] for v in self.base_concepts.values()]
        ret = []
        for i in range(n_concepts):
            c = cs[np.random.randint(0, len(cs), 1)[0]]
            l = np.random.randint(0, max_length - 1, 1)[0]
            for _ in range(l):
                o = np.random.randint(0, 3, 1)[0]
                if o == 0:
                    c = self.compose(c, c, 'inv')[1]
                elif o == 1:
                    c_ = cs[np.random.randint(0, len(cs), 1)[0]]
                    c = self.compose(c, c_, 'union')[1]
                else:
                    c_ = cs[np.random.randint(0, len(cs), 1)[0]]
                    c = self.compose(c, c_, 'inter')[1]
            ret.append(c)
        return ret

    def free(self):
        for k, [_, dic] in self.base_concepts.items():
            dic.clear()

    # prune concepts which can be removed
    def clean_concepts(self):
        cleaned = 0
        marked_for_removal = []
        for concept_name, [_, dic] in self.base_concepts.items():
            if self.check_clean(concept_name):
                marked_for_removal.append(concept_name)
                cleaned += 1

        for concept_name in marked_for_removal:
            del self.base_concepts[concept_name]

        print(f'Pruned: {cleaned} concepts')
        print(f'Remaining: {len(self.base_concepts)} concepts')

    # check if a concept should be removed. remove concepts if they are not present
    # at all.
    def check_clean(self, concept_name):
        dic = self.base_concepts[concept_name][1]
        total_nodes = 0
        active_count = 0
        for _, graph_mask in dic.items():
            active_count += graph_mask.node_mask.sum()
            total_nodes += graph_mask.node_mask.shape[0]
        proportion = active_count / total_nodes
        return proportion == 0 or proportion == 1

    # get the mask of a specific concept for a graph
    def get_mask(self, concept, graph_idx):
        return self.base_concepts[concept.name()][1][graph_idx]

    # get the number of base concepts
    def num_concepts(self):
        return len(self.base_concepts)

    # logically compose two concepts using an operator. this will compose all of the
    # graph masks which are being tracked by this class.
    def compose(self, concept_a, concept_b, operator):
        if operator == 'union':
            concept_c = concept_a.union(concept_b)
        elif operator == 'inter':
            concept_c = concept_a.inter(concept_b)
        elif operator == 'inv':
            concept_c = concept_a.inv()
        else:
            raise Exception()

        if concept_c.name() in self.base_concepts:
            return [None, concept_c]

        self.base_concepts[concept_c.name()][0] = concept_c

        masks_a = self.base_concepts[concept_a.name()][1]
        masks_b = self.base_concepts[concept_b.name()][1]

        # for graph_idx in masks_a.keys():
        #     self.base_concepts[concept_c][graph_idx] = concept_c(self.graphs[graph_idx], graph_idx)

        for graph_idx in masks_a.keys():
            mask_a = masks_a[graph_idx]
            mask_b = masks_b[graph_idx]

            mask_c = None

            if operator == 'union':
                mask_c = mask_a.union(mask_b)
            elif operator == 'inter':
                mask_c = mask_a.inter(mask_b)
            elif operator == 'inv':
                mask_c = mask_a.inv()

            self.base_concepts[concept_c.name()][1][graph_idx] = mask_c

        return [None, concept_c]

    # expand to the next length of the beam search.
    def expand(self):
        marked_for_deletion = defaultdict(int)

        # only keep candidates within the beam by removing all others.
        if self.cur_length > 1:
            for neuron in self.neuron_concept_tracker.keys():
                tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                    c.length() == self.cur_length]
                idxs1 = np.argsort([v[0] for [v, _] in tracked_concepts])

                for i in idxs1[-BEAM_WIDTH:]:
                    marked_for_deletion[tracked_concepts[i][1].name()] += 1

            deleted_set = set()
            for neuron in self.neuron_concept_tracker.keys():
                tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                    c.length() == self.cur_length]
                for _, c in tracked_concepts:
                    c_name = c.name()
                    if c.length() == self.cur_length and marked_for_deletion[c_name] == 0 and c_name not in deleted_set:
                        self.base_concepts[c_name][1].clear()
                        deleted_set.add(c_name)

            print(f'Deleted: {len(deleted_set)}')

        pruned = 0

        # expand to the next depth
        for neuron in self.neuron_concept_tracker.keys():
            tracked_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if
                                c.length() == self.cur_length]
            unary_concepts = [[v, c] for [v, c] in self.neuron_concept_tracker[neuron] if c.length() == 1]

            idxs1 = np.argsort([v[0] for [v, _] in tracked_concepts])[-BEAM_WIDTH:]
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

        self.cur_length += 1

        print(f'Pruned: {pruned} concepts!')

    # Returns the concept indexes for neuron_idx which have not yet been scored.
    def get_unscored(self, neuron_idx):
        ret = []
        for idx, (score, _) in enumerate(self.neuron_concept_tracker[neuron_idx]):
            if score is None:
                ret.append(idx)
        return ret

    def truth(self, neuron_idx, concept_idxs):
        """
        :param neuron_idx: neuron index
        :return: matrix of shape #concepts x #nodes (across all graphs)
        """
        mat = []
        for concept_idx in concept_idxs:
            [_, concept] = self.neuron_concept_tracker[neuron_idx][concept_idx]
            row = []
            for graph_idx, concept_mask in self.base_concepts[concept.name()][1].items():
                row.append(concept_mask.node_mask)
            mat.append(torch.cat(row).unsqueeze(0))
        return torch.cat(mat)

    def match(self, neuron_activations, norm, inds):
        """
        Updates score for each concept for all neurons.
        :param neuron_activations:  #neurons x #nodes, all graphs stacked, normalised
        :param norm: #nodes x 1, normalisation term
        """

        n_nodes = neuron_activations.shape[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        neuron_activations = neuron_activations.to(device)
        inds = inds.to(device)

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

            err = torch.zeros(n_concepts).to(device)
            targets = targets.to(device)

            [alpha, beta, gamma] = self.omega

            final_thresholds = torch.zeros(activations.shape[0], device=device)

            for thresh in np.arange(alpha, beta) / gamma * max_act.item():
                acts = activations > thresh

                msk = torch.zeros(activations.shape, device=device)
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

        del neuron_activations

        ret_dic = {}
        for neuron_idx in self.neuron_concept_tracker.keys():
            ret_dic[neuron_idx] = {}
            for [val, concept] in self.neuron_concept_tracker[neuron_idx]:
                ret_dic[neuron_idx][concept.name()] = (concept, val[0], val[1])
        return ret_dic

    def similarity(self, neuron_activations):
        """
        Computes concept similarity between all neurons and all concepts.
        :param neuron_activations: feature maps, shape should be #nodes x #neurons
        :return: #neurons x #concepts matrix of concept scores.
        """
        neuron_activations = neuron_activations.T
        n_neurons = neuron_activations.shape[0]

        n_a = neuron_activations.repeat_interleave(len(self.concepts), 0)
        mask = self.node_mask.repeat(n_neurons, 1)

        row_mins = n_a.min(dim=1)[0]
        row_maxs = n_a.max(dim=1)[0]
        denom = row_maxs - row_mins
        zero_rows = (n_a.abs().sum(axis=1).bool() == 0).nonzero().squeeze(1)

        denom = denom.index_fill(dim=0, index=zero_rows, value=1.0)
        normed = (n_a - row_mins[:, None]) / denom[:, None]

        err = torch.nn.MSELoss(reduction='none')(normed, mask)
        err = 1 - err.mean(axis=1)  # (1, n_nodes x n_concepts)

        return err.reshape(n_neurons, len(self.concepts))

    def self_expand(self):
        ors = []
        ands = []
        nots = []

        n_concepts = len(self.concepts)

        for i in range(n_concepts):
            for j in range(i, n_concepts):
                ors.append(self.concepts[i].union(self.concepts[j]))
                ands.append(self.concepts[i].inter(self.concepts[j]))
                nots.append(self.concepts[i].inv())

        self.concepts.extend(ors)
        self.concepts.extend(ands)
        self.concepts.extend(nots)

        return ConceptSet(*copy.deepcopy(self.concepts))


# experiment for concepts which are not binary values over the nodes but rather
# reals.
class PolynomialShaper(torch.nn.Module):
    def __init__(self, n_degrees, neuron_mat, concept_mat, graph_idxs, epochs=100):
        super().__init__()
        self.n_degrees = n_degrees
        self.neuron_mat = neuron_mat
        self.concept_mat = concept_mat
        self.n_concepts = concept_mat.shape[0]
        self.n_nodes = concept_mat.shape[1]
        self.graph_idxs = graph_idxs
        self.register_parameter('coefs', torch.nn.Parameter(-0.5 + torch.rand((self.n_concepts, self.n_degrees))))
        self.epochs = epochs

    def compute(self):
        t = torch.zeros((self.n_concepts, self.n_nodes), device=self.coefs.device)

        for deg in range(self.n_degrees):
            if deg == 0:
                t += self.coefs[:, 0].unsqueeze(1)
            else:
                t += self.neuron_mat.pow(deg) * self.coefs[:, deg].unsqueeze(1)

        t = (t - self.concept_mat).pow(2)  # (n_concepts, n_nodes)
        t = torch_scatter.scatter_add(t, self.graph_idxs)
        return t.mean(dim=1)  # (n_concepts, 1)

    def optimise(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.01)

        for _ in range(self.epochs):
            t = self.compute().sum()

            optim.zero_grad()
            t.backward()
            optim.step()


# a hack
def higher_order_wrapper(graph, lamb):
    adj_list = edge_index_to_adj_list(graph.edge_index, exclude_self=True)
    mask = lamb(graph).node_mask

    mask_set = set([i for i in range(mask.shape[0]) if mask[i] != 0])
    node_set = set()

    for u, neighs in adj_list.items():
        for v in neighs:
            if v in mask_set:
                node_set.add(u)
                break

    return ConceptMask(graph, set(), node_set)


# a concept object which can be applied to a graph as a function, returning a graph mask.
# concepts keep track of lambda functions.
class Concept:
    def __init__(self, lamb, children=[], operator=''):
        self.lamb = lamb
        self.concept_set = None
        self.group = None
        self.children = children
        self.operator = operator

        assert operator in ['OR', 'AND'] and len(children) == 2 or operator == 'NOT' and len(children) == 1 \
               or operator == '' and children == []

        self.children.sort(key=lambda x: x.name())

    def __call__(self, graph, graph_idx=None):
        return self.lamb(graph)

    def __str__(self):
        return f'Conc({self.name()})'

    def __repr__(self):
        return self.__str__()

    def name(self):
        if self.operator == 'NOT':
            return f'(NOT {self.children[0].name()})'
        return f'({self.children[0].name()} {self.operator} {self.children[1].name()})'

    def length(self):
        if len(self.children) == 1:  # negation
            return self.children[0].length()
        return sum([c.length() for c in self.children])

    def flatten(self):
        ret = [self]
        for child in self.children:
            ret.extend(child.flatten())
        return ret

    def concept_groups(self):
        ret = set()
        if self.group is not None:
            ret.add(self.group)
        for concept in self.children:
            ret = ret.union(concept.concept_groups())
        return ret

    def union(self, concept):
        return Concept(lambda g: self.__call__(g).union(concept(g)),
                       children=[self, concept],
                       operator='OR')

    def inter(self, concept):
        return Concept(lambda g: self.__call__(g).inter(concept(g)),
                       children=[self, concept],
                       operator='AND')

    def inv(self):
        return Concept(lambda g: self.__call__(g).inv(),
                       children=[self],
                       operator='NOT')


# a class for base concepts used for the concept search algorithm.
class BaseConcept(Concept):
    def __init__(self, name, lamb):
        super().__init__(lamb)
        self._name = name

    def name(self):
        return self._name

    def length(self):
        return 1


# a concept mask over a graph
class ConceptMask:
    def __init__(self, graph, edges, nodes):
        self.edges = edges
        self.nodes = nodes
        self.graph = graph

        self.node_mask = np.zeros(graph.x.shape[0])
        self.node_mask[list(self.nodes)] = 1
        self.node_mask = torch.FloatTensor(self.node_mask)

    def similarity(self, neuron_activations):
        denom = neuron_activations.max() - neuron_activations.min()
        if denom != 0:
            normed = (neuron_activations - neuron_activations.min()) / denom
        else:
            normed = neuron_activations
        return 1 - torch.nn.MSELoss()(normed, self.node_mask).item()

    def union(self, concept_mask):
        # new_name = '( ' + self.name + ' OR ' + concept_mask.name + ' )'
        edges_new = self.edges.union(concept_mask.edges)
        nodes_new = self.nodes.union(concept_mask.nodes)
        return ConceptMask(self.graph, edges_new, nodes_new)

    def inter(self, concept_mask):
        # new_name = '( ' + self.name + ' AND ' + concept_mask.name + ' )'
        edges_new = self.edges.intersection(concept_mask.edges)
        nodes_new = self.nodes.intersection(concept_mask.nodes)
        return ConceptMask(self.graph, edges_new, nodes_new)

    def inv(self):
        edge_tuples = set(edge_index_to_tuples(self.graph.edge_index))
        # new_name = f'(NOT {self.name})'
        edges_new = edge_tuples - self.edges
        nodes_new = set(list(np.arange(self.graph.x.shape[0]))) - self.nodes
        return ConceptMask(self.graph, edges_new, nodes_new)


# this class contains basic degree concept implementations.
class BasicConcepts:
    def degree(self, graph, k, operator='<'):
        """
        Nodes with degree in range.
        """
        adj_list = edge_index_to_adj_list(graph.edge_index)
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            if operator == '<' and len(adj_list[node_idx]) < k:
                node_set.add(node_idx)
            elif operator == '>' and len(adj_list[node_idx]) > k:
                node_set.add(node_idx)
            elif operator == '=' and len(adj_list[node_idx]) == k:
                node_set.add(node_idx)
        return ConceptMask(graph, set(), node_set)

    def feature(self, graph, j, k, operator='<'):
        node_set = set()
        for node_idx in range(graph.x.shape[0]):
            if operator == '<' and graph.x[node_idx, j] < k:
                node_set.add(node_idx)
            elif operator == '>' and graph.x[node_idx, j] > k:
                node_set.add(node_idx)
            elif operator == '=' and graph.x[node_idx, j] == k:
                node_set.add(node_idx)
        return ConceptMask(graph, set(), node_set)

    def neigh_degree(self, graph, k, operator='<', require=1, hops=1):
        """
        Nodes where neighbour satisfies degree condition.
        """
        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)

        mask = []
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            ns = []
            for neigh in adj_list[node_idx]:
                if operator == '<' and len(adj_list[neigh]) < k or operator == '>' and len(adj_list[neigh]) > k \
                        or operator == '=' and len(adj_list[neigh]) == k:
                    ns.append(neigh)
            if len(ns) >= require:
                for neigh in ns:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                    node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)

    def neigh_feature(self, graph, j, k, operator='<', require=1, hops=1):
        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)

        mask = []
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            ns = []
            for neigh in adj_list[node_idx]:
                if operator == '<' and graph.x[neigh, j] < k or operator == '>' and graph.x[neigh, j] > k \
                        or operator == '=' and graph.x[neigh, j] == k:
                    ns.append(neigh)
            if len(ns) >= require:
                for neigh in ns:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                    node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)


# contains concept implementations for molecular graphs.
class MoleculeConcepts:
    def __init__(self, labels):
        self.labels = labels
        self.labels_inv = {pair[1]: pair[0] for pair in self.labels.items()}
        self.n_atoms = len(self.labels)

    def is_atom(self, x, atom):
        if atom == 'X':
            return True
        return not (x != torch.nn.functional.one_hot(torch.tensor(self.labels[atom]), self.n_atoms)).any()

    def element(self, x):
        return self.labels_inv[torch.argmax(x, dim=0).item()]

    def is_element(self, graph, ele):
        node_set = set()

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele):
                node_set.add(node_idx)

        return ConceptMask(graph, set(), node_set)

    def AB_K(self, graph, a, b, k):
        adj_list = edge_index_to_adj_list(graph.edge_index)
        mask = []

        assert a in self.labels.keys()
        assert b in self.labels.keys()

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], a):
                ox_edges = []
                for neigh in adj_list[node_idx]:
                    if self.is_atom(graph.x[neigh], b) and len(adj_list[neigh]) == 1:
                        ox_edges.append((node_idx, neigh))
                        ox_edges.append((neigh, node_idx))
                if len(ox_edges) == 2 * k:
                    mask.extend(ox_edges)

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))

    def element_neighbour(self, graph, ele_self, *eles_nb, strict=True, hops=1):
        """
        All instances of atom being only connected to elements specified.
        """
        assert all([ele in self.labels.keys() for ele in eles_nb])

        adj_list = edge_index_to_adj_list(graph.edge_index, hops=hops, exclude_self=True)
        mask = []
        node_set = set()

        def is_subseq(x, y):
            it = iter(y)
            return all(c in it for c in x)

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele_self):
                neighbs = []
                for neigh in adj_list[node_idx]:
                    neighbs.append(self.element(graph.x[neigh]))
                str_a = ''.join(sorted(neighbs))
                str_b = ''.join(sorted(eles_nb))
                if strict and str_a != str_b:
                    continue
                if not strict and not is_subseq(str_b, str_a):
                    continue
                for neigh in adj_list[node_idx]:
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))
                node_set.add(node_idx)

        return ConceptMask(graph, set(mask), node_set)

    def element_same(self, graph, ele):
        """
        All edges connecting an element to the same element.
        """
        assert ele in self.labels.keys()

        edge_tuples = edge_index_to_tuples(graph.edge_index)
        mask = []

        carbon_indices = [node_idx for node_idx in range(graph.x.shape[0])
                          if self.is_atom(graph.x[node_idx], ele)]
        for idx in carbon_indices:
            for idx_ in carbon_indices:
                if idx != idx_ and (idx, idx_) in edge_tuples:
                    mask.append((idx, idx_))

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))

    def element_carbon(self, graph, ele):
        """
        All instances of atom being only connected to a carbon.
        """
        assert ele in self.labels.keys()

        adj_list = edge_index_to_adj_list(graph.edge_index)
        mask = []

        for node_idx in range(graph.x.shape[0]):
            if self.is_atom(graph.x[node_idx], ele) and len(adj_list[node_idx]) == 1:
                neigh = adj_list[node_idx][0]
                if self.is_atom(graph.x[neigh], 'C'):
                    mask.append((node_idx, neigh))
                    mask.append((neigh, node_idx))

        return ConceptMask(graph, set(mask), set(np.unique([[pair[0], pair[1]] for pair in mask])))


def concept_extract(model, dataset, task):
    from concept_utils import clean_concepts
    neuron_concepts = model.concept_search(task, dataset, depth=3, top=64, augment=False)
    cleaned_concepts, distilled = clean_concepts(neuron_concepts)
    return distilled


concepts_mutag = MoleculeConcepts({'C': 0, 'N': 1, 'O': 2, 'F': 3, 'I': 4, 'Cl': 5, 'Br': 6})
concepts_mutagenicity = MoleculeConcepts({'C': 0, 'O': 1, 'Cl': 2, 'H': 3, 'N': 4, 'F': 5, 'Br': 6,
                                          'S': 7, 'P': 8, 'I': 9, 'Na': 10, 'K': 11, 'Li': 12, 'Ca': 13})
concepts_prot = MoleculeConcepts({'A': 0, 'B': 1, 'C': 2})
concepts_nci = MoleculeConcepts({str(i): i for i in range(37)})

concepts_basic = BasicConcepts()
