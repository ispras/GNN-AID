from explainers.explainer import Explainer, finalize_decorator
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import copy
import shap

from explainers.NeuralAnalysis.our.concepts import ConceptSet
from explainers.NeuralAnalysis.orig.concept_utils import clean_concepts
from explainers.NeuralAnalysis.orig.graph_utils import edge_index_to_tuples, add_edge
from explainers.NeuralAnalysis.our.concept_ranker import by_weight
from data_structures.explanation import ConceptExplanationGlobal

from aux.utils import import_by_name

class NeuralAnalysisExplainer(Explainer):

    # TODO model.n_layers - layer can be reworked with _check_model_structure

    name = 'NeuralAnalysis'

    def __init__(self, gen_dataset, model, task, device):
        """

        Args:
            gen_dataset:
            model:
            task: specify the structure of dataset in order to generate base concepts
            depth: max base concepts in composite concept
            neuron_idxs: neurons have to be explained
            top: keeping only top neurons
            augment: whether augment dataset or not
            omega: ? ? ?
        """
        gen_dataset.train_test_split(percent_train_class=0.1)
        Explainer.__init__(self, gen_dataset, model)

        # if omega is None:
        #     omega = [10, 20, 20]
        # if neuron_idxs is None:
        #     neuron_idxs = []
        self.device = device
        self.task = task
        self.concepts = None

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        # TODO check if single-graph can be used
        return gen_dataset.is_multi()

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):


        # if 'level' not in kwargs:
        #     level = self.model.model_info['last_node_layer_ind']
        # else:
        #     level = kwargs['level']

        # modules = list(self.model.modules())
        # assert isinstance(modules[level], MessagePassing), 'Explained layer expected to be convolutional'

        if mode == 'global':
            if 'level' not in kwargs:
                pbar_n = (kwargs['depth'] - 1) * self.model.get_neurons()[self.model.model_info['last_graph_layer_ind'] - 1]
                self.pbar.reset(pbar_n)
            else:
                pbar_n = (kwargs['depth'] - 1) * self.model.get_neurons['level']
                self.pbar.reset(pbar_n)
            neuron_concepts = self.concept_search(**kwargs)
            cleaned_concepts, distilled = clean_concepts(neuron_concepts)  # QUE What is distilled ? ? ?
            if 'level' in kwargs or 'neuron_idxs' not in kwargs:
                self.raw_explanation = [cleaned_concepts, {}]
                for n in range(self.gen_dataset.num_classes):
                    neurons, vals = by_weight(self, n)
                    importance = [(x, str(y)) for x, y in zip(neurons, vals)]
                    for pair in importance:
                        if pair[0] not in self.raw_explanation[0]:
                            continue
                        else:
                            if pair[0] not in self.raw_explanation[1]:
                                self.raw_explanation[1][pair[0]] = {}
                            self.raw_explanation[1][pair[0]][n] = pair[1]
                self.concepts = self.raw_explanation.copy()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.pbar.n = pbar_n - 1 # TODO some kind of patch maybe fix needed
        self.pbar.update(1)
        self.pbar.close()

    def _finalize(self):
        kwargs = self._run_kwargs
        if self._run_mode == 'global':
            neuron_structure = self.model.get_neurons()
            if 'level' in kwargs or 'neuron_idxs' not in kwargs:  # TODO Refactor: Some kostyl code
                if 'level' not in kwargs:
                    level = self.model.model_info['last_graph_layer_ind'] - 1
                else:
                    level = kwargs['level']
                #self.explanation = ConceptExplanationGlobal(self.raw_explanation[level], neuron_structure[-level])
                self.explanation = ConceptExplanationGlobal(self.raw_explanation, neuron_structure[level])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Remove unpickable attributes
        self.pbar = None

    def concept_search(self, level=None, depth=1, neuron_idxs=None, top=64, augment=False, omega=None, **kwargs):

        # TODO neuron_idxs usage has to be implemented. Not used in original code but we can modify it

        if omega is None:
            omega = [10, 20, 20]

        assert depth >= 1 and ((level is None) or (neuron_idxs is None))
        if level is None and neuron_idxs is None:
            level = self.model.model_info['last_graph_layer_ind'] - 1

        dataset_aug = []
        if augment:
            print('Graph augmentation')

            for graph in tqdm(self.gen_dataset.dataset.index_select(self.gen_dataset.train_mask)):
                edge_tuples = edge_index_to_tuples(graph.edge_index)
                dataset_aug.append(copy.deepcopy(graph))
                for _ in range(4):
                    graph_new = copy.deepcopy(graph)
                    for _ in range(8):
                        node_i = np.random.randint(0, graph_new.x.shape[0])
                        node_j = np.random.randint(0, graph_new.x.shape[0])
                        add_edge(graph_new, node_i, node_j)
                    dataset_aug.append(graph_new)
            new_dataset = dataset_aug
        else:
            new_dataset = self.gen_dataset.dataset.index_select(self.gen_dataset.train_mask)

        n_graphs = len(new_dataset)
        print("TEST")
        concept_set = ConceptSet(new_dataset, self.task, omega=omega, pbar=self.pbar, device=self.device)
        print("TEST")

        print('Performing inference')
        neuron_activations = []
        graph_sizes = []
        graph_inds = []

        for i in tqdm(range(n_graphs)):
            graph = new_dataset[i]
            # feature_maps = model.partial_forward(graph.x.to(model.device), graph.edge_index.to(model.device),
            #                                     ret_layer=model.n_layers - level).detach().cpu().T

            feature_maps = self.model.get_all_layer_embeddings(x=graph.x, edge_index=graph.edge_index)[level].detach().cpu().T

            # # TEST
            # con = model.conn_dict[(model.n_layers - level, 3)][0]
            # con_pool = import_by_name(con['pool']['pool_type'],
            #                           ["torch_geometric.nn"])
            #
            # concept_layer = con_pool(feature_maps.T, torch.tensor([0], device=self.device))

            neuron_activations.append(feature_maps)

            graph_sizes.extend([graph.x.shape[0]] * graph.x.shape[0])
            graph_inds.extend([i] * graph.x.shape[0])

        print('Keeping only top neurons')
        neuron_activations = torch.cat(neuron_activations, 1)
        nrns_vals = (neuron_activations != 0).sum(axis=1)
        neuron_idxs = nrns_vals.argsort()
        non_zero_neuron_idxs = []
        for idx in neuron_idxs:
            if nrns_vals[idx] == 0:
                continue
            non_zero_neuron_idxs.append(idx)
        non_zero_neuron_idxs = torch.LongTensor(non_zero_neuron_idxs)
        neuron_idxs = non_zero_neuron_idxs
        neuron_activations = neuron_activations.index_select(0, neuron_idxs[-top:])

        print('Performing search')

        for i in range(depth):
            ret = concept_set.match(neuron_activations, torch.tensor(graph_sizes), torch.tensor(graph_inds))

            if i < depth - 1:
                print('Adding concepts')
                concept_set.expand()
                print('Number of concepts: ' + str(concept_set.num_concepts()))

        concept_set.free()

        ret_dic = {}
        for k, v in ret.items():
            ret_dic[neuron_idxs[k].item()] = v

        return ret_dic

    def expl_gp_neurons(self, graph, y, level, concepts=None, gamma=0.5, rank=1, debug=False,
                        res=300, mode='greedy',
                        cum=True, explore=False, pool='mean', aggr='sum', show_labels=True,
                        show_contribs=False, entropic=False, show_node_mask=False, edge_thresh=None, force=False,
                        sigma=1.0, scores=[], names=[], as_molecule=False):
        x = graph.x
        edge_index = graph.edge_index

        assert pool in ['mean', 'min', 'rand']
        assert mode in ['greedy', 'fair']

        from explainers.NeuralAnalysis.our import concept_ranker

        if entropic or show_contribs:
            vals_ent = concept_ranker.by_entropy(self.model, graph.x, graph.edge_index, y, epochs=200, beta=1, sort=False)

        if show_contribs:
            vals_abs = concept_ranker.by_weight_x_val(self.model, graph.x, graph.edge_index, y, sort=False)

        # self.include_top = False
        # concept_layer = self.forward(x.to(self.device), edge_index.to(self.device)).detach().cpu().numpy()
        # self.include_top = True

        feature_maps = self.model.get_all_layer_embeddings(x=graph.x, edge_index=graph.edge_index)[level].detach().cpu().T

        concept_layer = self.concept_layer(graph.x, graph.edge_index, level, feature_maps=feature_maps)

        edge_weights = defaultdict(float)
        edge_set = {(pair[0].item(), pair[1].item()) for pair in edge_index.T}
        node_values = np.zeros(feature_maps.shape[0])

        if mode == 'greedy':
            importance = self.neuron_importance(method='weights')[y].squeeze()
            if entropic:
                weights = vals_ent - 0.5
            else:
                weights = (importance * concept_layer)[0]

            ranking = weights.argsort()

            if type(rank) is list:
                units = [ranking[-r] for r in rank]
            elif type(rank) is set:
                units = list(rank)
            else:
                units = ranking[-rank:][::-1] if cum else [ranking[-rank]]
                units_ = []
                for u in units:
                    if weights[u] <= 0:
                        continue
                    units_.append(u)
                units = units_

            mask_set = []
            node_vals_set = []

            def iou(s1, s2):
                if len(s1) == 0 or len(s2) == 0:
                    return 1 if len(s1) == len(s2) else 0
                assert len(s1) > 0 and len(s2) > 0
                s1_s = set(s1)
                s2_s = set(s2)
                assert len(s1_s) == len(s1) and len(s2_s) == len(s2)
                return len(s1_s.intersection(s2_s)) / len(s1_s.union(s2_s))

            for unit in units:
                if concepts is not None:
                    print(concepts[unit])

                if type(sigma) is list:
                    mask = feature_maps[:, unit] > sigma[unit]
                else:
                    order = feature_maps[:, unit].argsort().flip(0)
                    mask = torch.zeros(feature_maps.shape[0])
                    denom = (10 * feature_maps[:, unit]).exp().sum()
                    probs = (10 * feature_maps[:, unit]).exp() / denom
                    tot = 0
                    for k in order:
                        tot += probs[k]
                        mask[k] = 1.0
                        if tot >= sigma:
                            break

                where = mask.nonzero().squeeze(1).numpy()

                edges = []
                vals = []
                node_vals = []

                for i in range(feature_maps.shape[0]):
                    node_vals.append(feature_maps[i, unit])

                for (i, j) in edge_set:
                    if i == j:
                        continue
                    if (explore and (i in where or j in where)) or (not explore and i in where and j in where):
                        v_i = feature_maps[i, unit].item()
                        v_j = feature_maps[j, unit].item()
                        if pool == 'mean':
                            r = 0.5 * v_i + 0.5 * v_j
                        elif pool == 'min':
                            r = min(v_i, v_j)

                        edges.append((i, j))
                        vals.append(r)

                if (len(edges) == 0 or sum(node_vals) == 0) and not force:
                    continue

                too_similar = False

                if len(edges) != 0:
                    vals, edges = zip(*sorted(zip(vals, edges), reverse=True))

                    for unit_, edges_, vals_ in mask_set:
                        if iou(edges_[:gamma], edges[:gamma]) > 0.5:
                            too_similar = True
                            break

                if not too_similar:
                    if names != [] and scores != []:
                        print(f'Unit: {unit}  Concept: {names[unit]}  Score: {scores[unit]}')
                        if show_contribs:
                            print(f'ABS: {vals_abs[unit]} ENT: {vals_ent[unit]}')
                    mask_set.append((unit, edges, vals))
                    node_vals_set.append(node_vals)

            if debug:
                print(f'Mask set contains {len(mask_set)} masks')
                print([u for u, _, _ in mask_set])

            for i, (unit, edges, vals) in enumerate(mask_set):

                factor = 1.0
                for j, ((u, v), val) in enumerate(zip(edges, vals)):
                    if j >= gamma:
                        break
                    if aggr == 'sum':
                        edge_weights[(u, v)] = val * factor + edge_weights[(u, v)]
                    elif aggr == 'max':
                        edge_weights[(u, v)] = max(val * factor, edge_weights[(u, v)])
                node_values += np.array(node_vals_set[i]) * factor

        elif mode == 'fair':
            importance = torch.clone(self.lin1.weight[y].squeeze()).cpu()
            node_weights = (feature_maps * importance).index_select(1, (importance > 0).nonzero().squeeze()).sum(axis=1)
            edge_list = [(pair[0].item(), pair[1].item()) for pair in edge_index.T]

            for (i, j) in edge_list:
                v_i = node_weights[i]
                v_j = node_weights[j]
                if pool == 'mean':
                    r = 0.5 * v_i + 0.5 * v_j
                elif pool == 'min':
                    r = min(v_i, v_j)
                edge_weights[(i, j)] = r

        final_mask = np.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge in edge_weights:
                final_mask[i] = edge_weights[edge]

        if edge_thresh is not None:
            inds = (final_mask <= edge_thresh).nonzero()[0]
            final_mask[(final_mask > edge_thresh).nonzero()[0]] = 1.0
            final_mask[inds] = 0.0

        return final_mask, node_values

    def neuron_importance(self, train_dataset=None, test_dataset=None, method='shap'):
        if method == 'shap':
            X_train = []
            X_test = []

            for data in train_dataset:
                data = data.to(self.device)
                neurons = self.concept_layer(data.x, data.edge_index, data.batch)
                X_train.append(neurons.detach())
            for data in test_dataset:
                data = data.to(self.device)
                neurons = self.forward(data.x, data.edge_index, data.batch)
                X_test.append(neurons.detach())

            X_train = torch.row_stack(X_train)
            X_test = torch.row_stack(X_test)

            print('DEBUG___')
            print('X_train shape: ' + str(X_train.shape))
            print('X_test shape: ' + str(X_test.shape))

            top_level = torch.nn.Sequential(
                self.lin1,
                torch.nn.Softmax()
            )

            explainer = shap.DeepExplainer(top_level, X_train)
            shap_values = explainer.shap_values(X_test)

            return X_train, X_test, shap_values
        elif method == 'weights':
            t = torch.clone(self.lin1.weight).detach().cpu()
            return [t[0].unsqueeze(0).numpy(), t[1].unsqueeze(0).numpy()]
        else:
            raise Exception('No method %s' % method)

    def concept_layer(self, x, edge_index, level, batch=None, feature_maps=None):
        # TODO kostyl have to be reworked with get_all_layer_embeddings

        con = self.model.conn_dict[(self.model.n_layers - level, 3)][0]
        con_pool = import_by_name(con['pool']['pool_type'],
                                  ["torch_geometric.nn"])
        if feature_maps is None:
            feature_maps = self.model.get_all_layer_embeddings(x=x, edge_index=edge_index)[level].detach().cpu().T
        concept_layer = con_pool(feature_maps.T, torch.tensor([0], device=self.device) if batch is None else batch)
        return concept_layer