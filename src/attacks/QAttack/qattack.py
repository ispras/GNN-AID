import copy
import math
import numpy as np
import random

from tqdm import tqdm
from attacks.evasion_attacks import EvasionAttacker
from attacks.QAttack.utils import get_adj_list, from_adj_list, adj_list_oriented_to_non_oriented

class QAttacker(EvasionAttacker):
    name = "QAttack"

    def __init__(self, population_size, individual_size, generations, prob_cross, prob_mutate, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.individual_size = individual_size
        self.generations = generations
        self.prob_cross = prob_cross
        self.prob_mutate = prob_mutate

    def init(self, gen_dataset):
        """
        Init first population:
            gen_dataset - graph-dataset
            population_size - size of population
            individual_size - amount of rewiring actions in one gene/individual
        """
        self.population = []

        self.adj_list = get_adj_list(gen_dataset)

        for i in tqdm(range(self.population_size), desc='Init first population:'):
            non_isolated_nodes = set(gen_dataset.dataset.edge_index[0].tolist()).union(
                set(gen_dataset.dataset.edge_index[1].tolist()))
            selected_nodes = np.random.choice(list(self.adj_list.keys()), size=self.individual_size, replace=False)
            gene = {}
            for n in selected_nodes:
                connected_nodes = set(self.adj_list[n])
                connected_nodes.add(n)
                addition_nodes = non_isolated_nodes.difference(connected_nodes)
                gene[n] = {'add': np.random.choice(list(addition_nodes), size=1),
                           'del': np.random.choice(list(self.adj_list[n]), size=1)}
            self.population.append(gene)

    def fitness(self, model, gen_dataset):
        """
        Calculate fitness function with node classification
        """

        fit_scores = []
        for i in range(self.population_size):
            # Get rewired dataset
            dataset = copy.deepcopy(gen_dataset.dataset)
            rewiring = self.population[i]
            adj_list = get_adj_list(dataset)
            for n in rewiring.keys():
                adj_list[n] = list(set(adj_list[n]).union({int(rewiring[n]['add'])}).difference({int(rewiring[n]['del'])}))
            dataset.edge_index = from_adj_list(adj_list)

            # Get labels from black-box
            labels = model.gnn.get_answer(dataset.x, dataset.edge_index)
            labeled_nodes = dict(enumerate(labels.tolist()))
            # labeled_nodes = {n: labels.tolist()[n-1] for n in adj_list.keys()}  # FIXME check order for labels and node id consistency

            # Calculate modularity
            Q = self.modularity(adj_list, labeled_nodes)
            fit_scores.append(1 / math.exp(Q))
        return fit_scores

    def fitness_individual(self, model, gen_dataset, gene):
        dataset = copy.deepcopy(gen_dataset.dataset)
        rewiring = gene
        adj_list = get_adj_list(dataset)
        for n in rewiring.keys():
            adj_list[n] = list(set(adj_list[n]).union(set(rewiring[n]['add'])).difference(set(rewiring[n]['del'])))
        dataset.edge_index = from_adj_list(adj_list)

        # Get labels from black-box
        labels = model.gnn.get_answer(dataset.x, dataset.edge_index)
        labeled_nodes = dict(enumerate(labels.tolist()))
        # labeled_nodes = {n: labels.tolist()[n-1] for n in adj_list.keys()}  # FIXME check order for labels and node id consistency

        # Calculate modularity
        Q = self.modularity(adj_list, labeled_nodes)
        return 1 / math.exp(Q)

    @staticmethod
    def modularity(adj_list, labeled_nodes):
        """
        Calculation of graph modularity with specified node partition on communities
        """
        # TODO implement oriented-modularity

        inc = dict([])
        deg = dict([])

        links = 0
        non_oriented_adj_list = adj_list_oriented_to_non_oriented(adj_list)
        for k, v in non_oriented_adj_list.items():
            links += len(v)
        if links == 0:
            raise ValueError("A graph without link has an undefined modularity")
        links //= 2

        for node, edges in non_oriented_adj_list.items():
            com = labeled_nodes[node]
            deg[com] = deg.get(com, 0.) + len(non_oriented_adj_list[node])
            for neighbor in edges:
                edge_weight = 1 # TODO weighted graph to be implemented
                if labeled_nodes[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.) + float(edge_weight)
                    else:
                        inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

        res = 0.
        for com in set(labeled_nodes.values()):
            res += (inc.get(com, 0.) / links) - \
                   (deg.get(com, 0.) / (2. * links)) ** 2
        return res

    def selection(self, model_manager, gen_dataset):
        fit_scores = self.fitness(model_manager, gen_dataset)
        probs = [i / sum(fit_scores) for i in fit_scores]
        selected_population = copy.deepcopy(self.population)
        for i in range(self.population_size):
            selected_population[i] = copy.deepcopy(self.population[np.random.choice(
                self.population_size, 1, False, probs)[0]])
        self.population = selected_population

    def crossover(self):
        for i in range(0, self.population_size // 2, 2):
            parent_1 = self.population[i]
            parent_2 = self.population[i + 1]
            crossover_prob = np.random.random()
            if crossover_prob <= self.prob_cross:
                self.population[i * 2], self.population[i * 2 + 1] = self.gene_crossover(parent_1, parent_2)
            else:
                self.population[i * 2], self.population[i * 2 + 1] = (copy.deepcopy(self.population[i * 2]),
                                                                      copy.deepcopy(self.population[i * 2 + 1]))

    def gene_crossover(self, parent_1, parent_2):
        parent_1_set = set(parent_1.keys())
        parent_2_set = set(parent_2.keys())

        parent_1_unique = parent_1_set.difference(parent_2_set)
        parent_2_unique = parent_2_set.difference(parent_1_set)

        parent_1_cross = list(parent_1_unique)
        parent_2_cross = list(parent_2_unique)

        assert len(parent_1_cross) == len(parent_2_cross)
        if len(parent_1_cross) == 0:
            return parent_1, parent_2
        n = np.random.randint(1, len(parent_1_cross) + 1)
        parent_1_cross = random.sample(parent_1_cross, n)
        parent_2_cross = random.sample(parent_2_cross, n)

        parent_1_set.difference_update(parent_1_cross)
        parent_2_set.difference_update(parent_2_cross)

        parent_1_set.update(parent_2_cross)
        parent_2_set.update(parent_1_cross)

        child_1 = {}
        child_2 = {}
        for n in parent_1_set:
            if n in parent_1.keys():
                child_1[n] = parent_1[n]
            else:
                child_1[n] = parent_2[n]
        for n in parent_2_set:
            if n in parent_2.keys():
                child_2[n] = parent_2[n]
            else:
                child_2[n] = parent_1[n]

        return child_1,child_2

    def mutation(self, gen_dataset):
        for i in range(self.population_size):
            keys = self.population[i].keys()
            for n in list(keys):
                mutation_prob = np.random.random()
                if mutation_prob <= self.prob_mutate:
                    mut_type = np.random.randint(3)
                    dataset = copy.deepcopy(gen_dataset.dataset)
                    rewiring = self.population[i]
                    adj_list = get_adj_list(dataset)
                    for n in rewiring.keys():
                        adj_list[n] = list(
                            set(adj_list[n]).union(set([int(rewiring[n]['add'])])).difference(set([int(rewiring[n]['del'])])))
                    dataset.edge_index = from_adj_list(adj_list)
                    non_isolated_nodes = set(gen_dataset.dataset.edge_index[0].tolist()).union(
                        set(gen_dataset.dataset.edge_index[1].tolist()))
                    non_drain_nodes = set(gen_dataset.dataset.edge_index[0].tolist())
                    if mut_type == 0:
                        # add mutation
                        connected_nodes = set(self.adj_list[n])
                        connected_nodes.add(n)
                        addition_nodes = non_isolated_nodes.difference(connected_nodes)
                        self.population[i][n]['add'] = np.random.choice(list(addition_nodes), 1)
                    elif mut_type == 1:
                        # del mutation
                        self.population[i][n]['del'] = np.random.choice(list(adj_list[n]), 1)
                    else:
                        selected_nodes = set(self.population[i].keys())
                        #non_selected_nodes = non_isolated_nodes.difference(selected_nodes)
                        non_drain_nodes = non_drain_nodes.difference(selected_nodes)
                        new_node = np.random.choice(list(non_drain_nodes), size=1, replace=False)[0]
                        self.population[i].pop(n)
                        addition_nodes = non_isolated_nodes.difference(set(self.adj_list[new_node]))
                        self.population[i][new_node] = {}
                        self.population[i][new_node]['add'] = np.random.choice(list(addition_nodes), 1)
                        self.population[i][new_node]['del'] = np.random.choice(list(adj_list[new_node]), 1)

    def elitism(self, model, gen_dataset):
        fit_scores = list(enumerate(self.fitness(model, gen_dataset)))
        fit_scores = sorted(fit_scores, key=lambda x: x[1])
        sort_order = [x[0] for x in fit_scores]
        self.population = [self.population[i] for i in sort_order]
        elitism_size = int(0.1 * self.population_size)
        self.population[:elitism_size] = self.population[-elitism_size:]
        return self.population[-1]


    def attack(self, model_manager, gen_dataset, mask_tensor):
        self.init(gen_dataset)

        for i in tqdm(range(self.generations), desc='Attack iterations:', position=0, leave=True):
            self.selection(model_manager, gen_dataset)
            self.crossover()
            self.mutation(gen_dataset)
            best_offspring = self.elitism(model_manager, gen_dataset)

        rewiring = best_offspring
        adj_list = get_adj_list(gen_dataset)
        for n in rewiring.keys():
            adj_list[n] = list(
                set(adj_list[n]).union(set([int(rewiring[n]['add'])])).difference(set([int(rewiring[n]['del'])])))

        gen_dataset.dataset.data.edge_index = from_adj_list(adj_list)
        return gen_dataset