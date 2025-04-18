from typing import Callable, Optional
from typing import List, Tuple, Dict
import torch
import abc
from tqdm import tqdm
from dig.xgraph.method.shapley import MarginalSubgraphDataset

from torch import Tensor

from explainers.explainer import Explainer, finalize_decorator
from explainers.explanation import AttributionExplanation


# FIXME Monkey Patch for SubgraphX until DIG library doesn't support torch-geometric 2.3.1
# PATCH BEGIN
MarginalSubgraphDataset.__abstractmethods__ = frozenset()
# PATCH END

from dig.xgraph.method import MCTS
from dig.xgraph.method.shapley import \
    gnn_score, sparsity
from dig.xgraph.method.subgraphx import reward_func, MCTSNode, find_closest_node_result

def GnnNetsGC2valueFunc(model, target_class):
    def value_func(batch):
        with torch.no_grad():
            # logits = model(data=batch)
            # probs = F.softmax(logits, dim=-1)
            probs = model.get_predictions(data=batch)
            score = probs[:, target_class]
        return score
    return value_func


def GnnNetsNC2valueFunc(model, node_idx, target_class):
    def value_func(data):
        with torch.no_grad():
            # logits = model(data=data)
            # probs = F.softmax(logits, dim=-1)
            probs = model.get_predictions(data=data)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = data.batch.max() + 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score
    return value_func


class _MCTS(MCTS):
    def __init__(self, pbar, *args, **kwargs):
        super(_MCTS, self).__init__(*args, **kwargs)
        self.pbar = pbar

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")
            self.pbar.update(1)

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


# TODO misha divide to orig SubgraphX and our
class SubgraphXExplainer(Explainer):
    r"""
    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.

    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_classes(:obj:`int`): Number of classes for the datasets
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        explain_graph(:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)
        save_dir(:obj:`str`, :obj:`None`): Root directory to save the explanation results (default: :obj:`None`)
        filename(:obj:`str`): The filename of results
        vis(:obj:`bool`): Whether to show the visualization (default: :obj:`True`)
    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphXExplainer(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)
    """
    name = 'SubgraphX'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        return\
            {'get_num_hops', 'get_predictions'}.issubset(dir(model_manager.gnn))

    def __init__(self, gen_dataset, model, device,
                 verbose: bool = False,
                 rollout: int = 20, min_atoms: int = 5,
                 c_puct: float = 10.0,
                 expand_atoms=14, high2low=False, local_radius=4, sample_num=100,
                 reward_method='mc_l_shapley',
                 subgraph_building_method='zero_filling', save_dir: Optional[str] = None,
                 filename: str = 'example', vis: bool = True):
        Explainer.__init__(self, gen_dataset, model)

        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        self.num_classes = gen_dataset.num_classes
        self.num_hops = model.get_num_hops()
        self.explain_graph = gen_dataset.is_multi()
        self.verbose = verbose

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

        self.explained_node = None
        self.explained_graph = None
        self.max_nodes = None
        # saving and visualization
        self.vis = vis
        # self.save_dir = save_dir
        self.filename = filename
        # self.save = True if self.save_dir is not None else False

    def get_reward_func(self, value_func, node_idx=None):
        if self.explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           node_idx=node_idx,
                           local_radius=self.local_radius,
                           sample_num=self.sample_num,
                           subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, x, edge_index, node_idx: int = None, score_func: Callable = None):
        if self.explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return _MCTS(self.pbar, x, edge_index,
                     node_idx=node_idx,
                     device=self.device,
                     score_func=score_func,
                     num_hops=self.num_hops,
                     n_rollout=self.rollout,
                     min_atoms=self.min_atoms,
                     c_puct=self.c_puct,
                     expand_atoms=self.expand_atoms,
                     high2low=self.high2low)

    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in
                        MCTSInfo_list]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = []
            for single_label_MCTSInfo_list in MCTSInfo_list:
                single_label_ret_list = [MCTSNode(device=self.device).load_info(node_info) for
                                         node_info in single_label_MCTSInfo_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        elif isinstance(MCTSNode_list[0][0], MCTSNode):
            ret_list = []
            for single_label_MCTSNode_list in MCTSNode_list:
                single_label_ret_list = [node.info for node in single_label_MCTSNode_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def explain(self, x: Tensor, edge_index: Tensor, label: int,
                max_nodes: int = 5,
                node_idx: Optional[int] = None,
                saved_MCTSInfo_list: Optional[List[List]] = None):
        # probs = self.model(x, edge_index).squeeze().softmax(dim=-1)
        probs = self.model.get_predictions(x, edge_index).squeeze()
        if self.explain_graph:
            if saved_MCTSInfo_list:
                results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

            if not saved_MCTSInfo_list:
                value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
                payoff_func = self.get_reward_func(value_func)
                self.mcts_state_map = self.get_mcts_class(x, edge_index, score_func=payoff_func)
                results = self.mcts_state_map.mcts(verbose=self.verbose)

            # l sharply score
            value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
            tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)

        else:
            if saved_MCTSInfo_list:
                results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

            self.mcts_state_map = self.get_mcts_class(x, edge_index, node_idx=node_idx)
            self.new_node_idx = self.mcts_state_map.new_node_idx
            # mcts will extract the subgraph and relabel the nodes
            value_func = GnnNetsNC2valueFunc(self.model,
                                             node_idx=self.mcts_state_map.new_node_idx,
                                             target_class=label)

            if not saved_MCTSInfo_list:
                payoff_func = self.get_reward_func(value_func,
                                                   node_idx=self.mcts_state_map.new_node_idx)
                self.mcts_state_map.set_score_func(payoff_func)
                results = self.mcts_state_map.mcts(verbose=self.verbose)

            tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)

        # keep the important structure
        masked_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                            if node in tree_node_x.coalition]

        # remove the important structure, for node_classification,
        # remain the node_idx when remove the important structure
        maskout_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                             if node not in tree_node_x.coalition]
        if not self.explain_graph:
            maskout_node_list += [self.new_node_idx]

        masked_score = gnn_score(masked_node_list,
                                 tree_node_x.data,
                                 value_func=value_func,
                                 subgraph_building_method=self.subgraph_building_method)

        maskout_score = gnn_score(maskout_node_list,
                                  tree_node_x.data,
                                  value_func=value_func,
                                  subgraph_building_method=self.subgraph_building_method)

        sparsity_score = sparsity(masked_node_list, tree_node_x.data,
                                  subgraph_building_method=self.subgraph_building_method)

        results = self.write_from_MCTSNode_list(results)
        related_pred = {'masked': masked_score,
                        'maskout': maskout_score,
                        'origin': probs[node_idx, label].item(),
                        'sparsity': sparsity_score}

        return results, related_pred

    def __call__(self, x: Tensor, edge_index: Tensor, **kwargs) \
            -> Tuple[None, List, List[Dict]]:
        r""" explain the GNN behavior for the graph using SubgraphX method
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - node_idx (:obj:`int`, :obj:`None`): The target node index when explain node classification task
                - max_nodes (:obj:`int`, :obj:`None`): The number of nodes in the final explanation results
        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        self.explained_node = node_idx = kwargs.get('node_idx')
        self.max_nodes = max_nodes = kwargs.get('max_nodes')  # default max subgraph size

        # collect all the class index
        labels = tuple(label for label in range(self.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        related_preds = []
        explanation_results = []
        saved_results = None
        # if self.save:
        #     if os.path.isfile(os.path.join(self.save_dir, f"{self.filename}.pt")):
        #         saved_results = torch.load(os.path.join(self.save_dir, f"{self.filename}.pt"))

        for label_idx, label in enumerate(ex_labels):
            results, related_pred = self.explain(x, edge_index,
                                                 label=label,
                                                 max_nodes=max_nodes,
                                                 node_idx=node_idx,
                                                 saved_MCTSInfo_list=saved_results)
            related_preds.append(related_pred)
            explanation_results.append(results)

        # if self.save:
        #     torch.save(explanation_results,
        #                os.path.join(self.save_dir, f"{self.filename}.pt"))

        return None, explanation_results, related_preds

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "local"
        idx = kwargs.pop('element_idx')
        self.pbar.reset(total=self.rollout * self.num_classes)
        if self.gen_dataset.is_multi():
            self.explained_graph = idx
            kwargs['node_idx'] = None  # default value 0 leads to an error
            data = self.gen_dataset.dataset.get(idx)
            _, self.raw_explanation, _ = self(data.x, data.edge_index, **kwargs)
        else:
            _, self.raw_explanation, _ = self(
                self.gen_dataset.data.x, self.gen_dataset.data.edge_index,
                node_idx=idx, **kwargs)
        self.pbar.close()

    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"
        self.explanation = AttributionExplanation(local=mode, nodes="binary", edges="binary")

        if self.gen_dataset.is_multi():
            data = self.gen_dataset.dataset.get(self.explained_graph)
            pred = self.model.get_answer(data.x, data.edge_index).item()
        else:
            data = self.gen_dataset.dataset.data
            pred = self.model.get_answer(data.x, data.edge_index)[self.explained_node].item()

        # Getting important nodes
        from dig.xgraph.method.subgraphx import find_closest_node_result
        if self.gen_dataset.is_multi():
            explanation_results = self.read_from_MCTSInfo_list(self.raw_explanation[pred])
            tree_node_x = find_closest_node_result(explanation_results, max_nodes=self.max_nodes)
            _nodes = tree_node_x.coalition  # TODO misha check correctness with different datasets
        else:
            explanation_results = self.read_from_MCTSInfo_list(self.raw_explanation[pred])
            tree_node_x = find_closest_node_result(explanation_results, max_nodes=self.max_nodes)
            mapping = {k: int(v) for k, v in enumerate(self.mcts_state_map.subset)}
            tree_node_x.coalition = [mapping[k] for k in tree_node_x.coalition]
            _nodes = tree_node_x.coalition

        # Edges
        # TODO misha can we simplify and avoid converting whole graph to networkx at each call?
        #  same for other such cases

        # Create undirected subgraph induced on important nodes
        nodes = set(_nodes)
        edges_values = {}
        for i, j in zip(*data.edge_index):
            i = int(i)
            j = int(j)
            if i in nodes and j in nodes:
                edges_values[f"{i},{j}"] = 1

        if self.gen_dataset.is_multi():
            edges_values = {self.explained_graph: edges_values}

        self.explanation.add_edges(edges_values)

        # Nodes
        nodes_values = {int(x): 1 for x in _nodes}
        if self.gen_dataset.is_multi():
            nodes_values = {self.explained_graph: nodes_values}

        self.explanation.add_nodes(nodes_values)

        # Remove unpickable attributes
        self.pbar = None
        self.mcts_state_map.pbar = None
