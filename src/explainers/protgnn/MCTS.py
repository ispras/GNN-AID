import math
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from functools import partial
from collections import Counter

class mcts_args():
    rollout: int = 10                         # the rollout number
    high2low: bool = False                    # expand children with different node degree ranking method
    c_puct: float = 5                         # the exploration hyper-parameter
    min_atoms: int = 4 #2 for NO2, 4-5 for C6                        # for the synthetic dataset, change the minimal atoms to 5.
    max_atoms: int = 6 #4 for NO2, 8-10 for C6
    expand_atoms: int = 10                     # # of atoms to expand children

class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def mcts_rollout(tree_node, state_map, data, graph, score_func):
    cur_graph_coalition = tree_node.coalition
    if len(cur_graph_coalition) <= mcts_args.min_atoms:
        return tree_node.P

    # Expand if this node has never been visited
    if len(tree_node.children) == 0:
        node_degree_list = list(graph.subgraph(cur_graph_coalition).degree)
        node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=mcts_args.high2low)
        all_nodes = [x[0] for x in node_degree_list]

        if len(all_nodes) < mcts_args.expand_atoms:
            expand_nodes = all_nodes
        else:
            expand_nodes = all_nodes[:mcts_args.expand_atoms]

        for each_node in expand_nodes:
            # for each node, pruning it and get the remaining sub-graph
            # here we check the resulting sub-graphs and only keep the largest one
            subgraph_coalition = [node for node in all_nodes if node != each_node]

            subgraphs = [graph.subgraph(c)
                         for c in nx.connected_components(graph.subgraph(subgraph_coalition))]
            main_sub = subgraphs[0]
            for sub in subgraphs:
                if sub.number_of_nodes() > main_sub.number_of_nodes():
                    main_sub = sub

            new_graph_coalition = sorted(list(main_sub.nodes()))

            """
            if new_graph_coalition == [0,1,2,3,6,7,11]:
                a = 6

            if not nx.is_connected(main_sub):
                print("ERROR")
            """

            # check the state map and merge the same sub-graph
            Find_same = False
            for old_graph_node in state_map.values():
                if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                    new_node = old_graph_node
                    Find_same = True

            if Find_same == False:
                new_node = MCTSNode(new_graph_coalition, data=data, ori_graph=graph)
                state_map[str(new_graph_coalition)] = new_node

            Find_same_child = False
            for cur_child in tree_node.children:
                if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                    Find_same_child = True

            if Find_same_child == False:
                tree_node.children.append(new_node)

        scores = compute_scores(score_func, tree_node.children)
        for child, score in zip(tree_node.children, scores):
            child.P = score

    sum_count = sum([c.N for c in tree_node.children])
    selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, data, graph, score_func)
    selected_node.W += v
    selected_node.N += 1
    return v


def mcts(data, gnnNet, prototype):
    data = Data(x=data.x, edge_index=data.edge_index)
    graph = to_networkx(data, to_undirected=True)
    data = Batch.from_data_list([data])
    num_nodes = graph.number_of_nodes()
    root_coalition = sorted([i for i in range(num_nodes)])
    root = MCTSNode(root_coalition, data=data, ori_graph=graph)
    state_map = {str(root.coalition): root}
    score_func = partial(gnn_prot_score, data=data, gnnNet=gnnNet, prototype=prototype)
    for rollout_id in range(mcts_args.rollout):
        mcts_rollout(root, state_map, data, graph, score_func)

    explanations = [node for _, node in state_map.items()]
    explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
    explanations = sorted(explanations, key=lambda x: len(x.coalition))

    result_node = explanations[0]
    for result_idx in range(len(explanations)):
        x = explanations[result_idx]
        if (len(x.coalition) <= mcts_args.max_atoms and x.P > result_node.P) or (len(result_node.coalition) < mcts_args.min_atoms and len(x.coalition) <= mcts_args.max_atoms):
            result_node = x

    # compute the projected prototype to return
    mask = torch.zeros(data.num_nodes).type(torch.float32)
    mask[result_node.coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)
    ret_edge_index = data.edge_index
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    # mask_data = mask_data.to(gnnNet.device)
    emb = gnnNet.get_all_layer_embeddings(data=mask_data, protgnn_plus=False)[
        int(gnnNet.prot_layer_name.split('_')[1]) - 1]
    return result_node.coalition, result_node.P, emb


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition)
        else:
            score = child.P
        results.append(score)
    return results


def gnn_prot_score(coalition, data, gnnNet, prototype):
    """ the similarity value of subgraph with selected nodes """
    epsilon = 1e-4
    mask = torch.zeros(data.num_nodes).type(torch.float32)
    mask[coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)
    ret_edge_index = data.edge_index
    #row, col = data.edge_index
    #edge_mask = (mask[row] == 1) & (mask[col] == 1)
    #ret_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    # mask_data = mask_data.to(gnnNet.device)
    # TODO Kirill, make this better
    emb = gnnNet.get_all_layer_embeddings(data=mask_data, protgnn_plus=False)[
        int(gnnNet.prot_layer_name.split('_')[1]) - 1]
    distance = torch.norm(emb-prototype)**2
    similarity = torch.log((distance+1) / (distance + epsilon))
    return similarity.item()