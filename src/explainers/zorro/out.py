from numpy import nan, inf
from numpy import nan, inf
from torch import is_tensor, no_grad, ones
from torch import zeros_like, zeros, cat
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from explainers.zorro.original import Zorro
from explainers.explainer import Explainer, finalize_decorator
from explainers.explanation import AttributionExplanation


class ZorroExplainer(Explainer, Zorro):
    name = 'Zorro'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        return\
            not gen_dataset.is_multi() and\
            {'get_num_hops', 'flow', 'get_answer'}.issubset(dir(model_manager.gnn))

    def __init__(self, gen_dataset, model, device, greedy=True, add_noise=False, samples=100):
        Zorro.__init__(self, model, device, log=False, record_process_time=False,
                       greedy=greedy, add_noise=add_noise, samples=samples)
        Explainer.__init__(self, gen_dataset, model)
        self.explained_node = None

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.model.get_num_hops(), edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.model.flow())

        x = x[subset]
        for key, item in kwargs:
            if is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "local"
        assert not self.gen_dataset.is_multi()
        idx = kwargs.pop('element_idx')
        self.pbar.reset(total=0)  # TODO misha num of steps = ???
        self.raw_explanation = self.explain_node(
            full_feature_matrix=self.gen_dataset.data.x,
            edge_index=self.gen_dataset.data.edge_index, node_idx=idx, **kwargs)
        # self.pbar.update(1)
        self.pbar.close()
        # Remove unpickable attributes
        self.pbar = None

    def explain_node(self, node_idx, full_feature_matrix, edge_index, tau=0.15, recursion_depth=inf,
                     save_initial_improve=False):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.explained_node = node_idx

        if save_initial_improve:
            self.initial_node_improve = [nan]
            self.initial_feature_improve = [nan]

        if hasattr(self.model, 'eval'):
            self.model.eval()

        if recursion_depth <= 0:
            self.logger.warning("Recursion depth not positve " + str(recursion_depth))
            raise ValueError("Recursion depth not positve " + str(recursion_depth))

        self.logger.info("------ Start explaining node " + str(node_idx))
        self.logger.debug("Distortion drop (tau): " + str(tau))
        self.logger.debug("Distortion samples: " + str(self.distortion_samples))
        self.logger.debug("Greedy variant: " + str(self.greedy))
        if self.greedy:
            self.logger.debug("Greediness: " + str(self.greediness))
            self.logger.debug("Ensure improvement: " + str(self.ensure_improvement))

        num_edges = edge_index.size(1)

        (num_nodes, self.num_features) = full_feature_matrix.size()

        self.full_feature_matrix = full_feature_matrix

        # Only operate on a k-hop subgraph around `node_idx`.
        self.computation_graph_feature_matrix, self.computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
            self.__subgraph__(node_idx, full_feature_matrix, edge_index)

        if self.add_noise:
            self.full_feature_matrix = cat(
                [self.full_feature_matrix, zeros_like(self.full_feature_matrix)],
                dim=0)

        self.node_idx = mapping

        self.num_computation_graph_nodes = self.computation_graph_feature_matrix.size(0)

        # Get the initial prediction.
        with no_grad():
            predicted_labels = self.model.get_answer(x=self.computation_graph_feature_matrix,
                                    edge_index=self.computation_graph_edge_index)
            # log_logits = self.model(x=self.computation_graph_feature_matrix,
            #                         edge_index=self.computation_graph_edge_index)
            # predicted_labels = log_logits.argmax(dim=-1)

            self.predicted_label = predicted_labels[mapping]

            # self.__set_masks__(computation_graph_feature_matrix, edge_index)
            self.to(self.computation_graph_feature_matrix.device)

            # self.pbar.total=int(self.num_computation_graph_nodes * self.num_features)
            # self.pbar.update(0)

            # if self.log:  # pragma: no cover
            #     self.overall_progress_bar = tqdm(total=int(self.num_computation_graph_nodes * self.num_features),
            #                                      position=1)
            #     self.overall_progress_bar.set_description(f'Explain node {node_idx}')

            possible_nodes = ones((1, self.num_computation_graph_nodes), device=self.device)
            possible_features = ones((1, self.num_features), device=self.device)

            self.selected_nodes = zeros((1, self.num_computation_graph_nodes), device=self.device)
            self.selected_features = zeros((1, self.num_features), device=self.device)

            initial_distortion = self.distortion()

            # safe the unmasked distortion
            self.logger.debug("Initial distortion without any mask: " + str(initial_distortion))

            if initial_distortion >= 1 - tau:
                # no mask needed, global distribution enough, see node 1861 in cora_GINConv
                self.pbar.update(self.pbar.total)

                self.logger.info("------ Finished explaining node " + str(node_idx))
                self.logger.debug("# Explanations: Select any nodes and features")
                if save_initial_improve:
                    return [
                               (self.selected_nodes.cpu().numpy(),
                                self.selected_features.cpu().numpy(),
                                [[nan, nan, initial_distortion], ]
                                )
                           ], None, None
                else:
                    return [
                        (self.selected_nodes.cpu().numpy(),
                         self.selected_features.cpu().numpy(),
                         [[nan, nan, initial_distortion], ]
                         )
                    ]
            else:
                self.epoch = 1
                minimal_nodes_and_features_sets = self.recursively_get_minimal_sets(
                    initial_distortion,
                    tau,
                    possible_nodes,
                    possible_features,
                    recursion_depth=recursion_depth,
                    save_initial_improve=save_initial_improve,
                )

            # if self.log:  # pragma: no cover
            #     self.overall_progress_bar.close()

        self.logger.info("------ Finished explaining node " + str(node_idx))
        self.logger.debug("# Explanations: " + str(len(minimal_nodes_and_features_sets)))

        if save_initial_improve:
            return minimal_nodes_and_features_sets, self.initial_node_improve, self.initial_feature_improve
        else:
            return minimal_nodes_and_features_sets

    def argmax_distortion_general_full(self,
                                       previous_distortion,
                                       possible_elements,
                                       selected_elements,
                                       save_all_pairs=False,
                                       **distortion_kwargs,
                                       ):
        best_element = None
        best_distortion_improve = -1000

        remaining_nodes_to_select = possible_elements - selected_elements
        num_remaining = remaining_nodes_to_select.sum()

        # if no node left break
        if num_remaining == 0:
            return best_element, best_distortion_improve

        self.pbar.total += int(num_remaining)
        self.pbar.update(0)
        # if self.log:  # pragma: no cover
        #     pbar = tqdm(total=int(num_remaining), position=0)
        #     pbar.set_description(f'Argmax {best_element}, {best_distortion_improve}')

        all_calculated_pairs = []

        i = 0
        while num_remaining > 0:
            if selected_elements[0, i] == 0 and possible_elements[0, i] == 1:
                num_remaining -= 1

                selected_elements[0, i] = 1

                distortion_improve = self.distortion(**distortion_kwargs) \
                                     - previous_distortion

                selected_elements[0, i] = 0

                if save_all_pairs:
                    all_calculated_pairs.append((i, distortion_improve))

                if distortion_improve > best_distortion_improve:
                    best_element = i
                    best_distortion_improve = distortion_improve
                    # if self.log:  # pragma: no cover
                    #     pbar.set_description(f'Argmax {best_element}, {best_distortion_improve}')

                # if self.log:  # pragma: no cover
                #     pbar.update(1)
                self.pbar.update(1)
            i += 1

        # if self.log:  # pragma: no cover
        #     pbar.close()
        if save_all_pairs:
            return best_element, best_distortion_improve, all_calculated_pairs
        else:
            return best_element, best_distortion_improve

    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"
        self.explanation = AttributionExplanation(local=mode, nodes="binary", features="binary")

        selected_nodes, selected_features, executed_selections = self.raw_explanation[0]

        # Features
        self.explanation.add_features(dict(enumerate(
            selected_features[0].astype(int).tolist())))

        # Nodes
        # Get nodes of the neighborhood of specific degree, according to edge directions
        ins = self.gen_dataset.data.edge_index[0]
        outs = self.gen_dataset.data.edge_index[1]
        neighborhood = {self.explained_node}
        for d in range(self.model.get_num_hops()):
            # Add all nodes from which we can reach current neighborhood(d)
            next_layer = set()
            for ix, j in enumerate(outs):
                j = int(j)
                if j in neighborhood:
                    next_layer.add(int(ins[ix]))
            neighborhood.update(next_layer)

        neighborhood = sorted(neighborhood)
        assert len(neighborhood) == len(selected_nodes[0])
        self.explanation.add_nodes(dict(
            zip(neighborhood, selected_nodes[0].astype(int).tolist())))
