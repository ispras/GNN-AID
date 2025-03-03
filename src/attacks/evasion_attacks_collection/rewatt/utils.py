from inspect import Parameter

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import random
from torch.distributions import Categorical
from torch_geometric.utils import k_hop_subgraph


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class EdgeRepresenter:
    def __init__(self, method='sum'):
        super(EdgeRepresenter, self).__init__()
        self.h = self._get_h_function(method)   # h() function from article

    def __call__(self, v1_emb, v2_emb, graph_emb):
        return torch.cat((graph_emb, self.h(v1_emb, v2_emb)), dim=-1)

    @staticmethod
    def _get_h_function(method):
        if method == "sum":
            return lambda a, b: a + b
        elif method == "mul":
            return lambda a, b: a * b
        elif method == "max":
            return lambda a, b: torch.where(a.norm(dim=-1) > b.norm(dim=-1), a, b)
        else:
            raise ValueError("Unsupported method")


class GraphRepresenter:
    def __init__(self, method='mean'):
        super(GraphRepresenter, self).__init__()
        self.method = method

    def __call__(self, node_embeddings):
        if self.method == 'mean':
            return torch.mean(node_embeddings, dim=0)
        elif self.method == 'max':
            return torch.max(node_embeddings, dim=0)[0]
        else:
            raise ValueError("Unsupported pooling method")


def vc_representation(vc_emb, v_et1):
    """
    Parameters:
    - v1_emb: The embedding of v_fir node.
    - edge_repr: Representation of an edge in the context of nodes v_fir and v_sec.
    - vc_emb: The embedding of v_c node.
    """
    return torch.cat((v_et1, vc_emb), dim=-1)


class GraphState:
    def __init__(self, x, edge_index, y):
        self.x = x
        self.edge_index = edge_index
        self.y = y


class GraphEnvironment:
    def __init__(self, gnn_model, initial_state: GraphState, eps, node_idx=None):
        """
        Класс окружения, которое принимает действия агента и обновляет состояние.

        :param initial_state: Начальное состояние графа (объект GraphState).
        """
        self.gnn_model = gnn_model
        self.K = int(eps * initial_state.edge_index.size(1))
        self.initial_state = initial_state
        self.current_state = initial_state
        if node_idx is not None:
            self.node_idx = node_idx
            self.graph_classification_task = False
        else:
            self.graph_classification_task = True

    def step(self, action):
        """
        Выполняет действие, изменяет состояние и возвращает новое состояние и награду.

        :param action: Кортеж (vfir, vsec, vthi) - операция переподключения рёбер.
        :return: (новое состояние, награда)
        """
        new_state = self.apply_rewiring(self.current_state, action)  # Изменение графа
        reward = self.calculate_reward(new_state)  # Вычисление награды
        self.current_state = new_state  # Обновляем текущее состояние
        return new_state, reward

    def apply_rewiring(self, state, action):
        """
        Применяет операцию переподключения рёбер к текущему состоянию.

        :param state: Текущее состояние графа.
        :param action: (vfir, vsec, vthi) - узлы, участвующие в переподключении.
        :return: Новое состояние графа (GraphState).
        """
        v_fir, v_sec, v_thi = action

        # Копируем рёбра, чтобы не менять исходные данные
        edge_index_new = state.edge_index.clone()

        # Ищем ребро, которое нужно удалить
        mask = (edge_index_new[0] == v_fir) & (edge_index_new[1] == v_sec)

        # Удаляем ребро (v_fir, v_sec) и добавляем (v_thi, v_fir) !!!
        edge_index_new = torch.cat([
            edge_index_new[:, ~mask],  # Оставляем только те рёбра, которые НЕ удаляются
            torch.tensor([[v_thi], [v_fir]], dtype=torch.long)  # Добавляем новое ребро
        ], dim=1)

        if self.graph_classification_task:
            y = self.gnn_model(state.x, edge_index_new).argmax()
        else:
            y = self.gnn_model(state.x, edge_index_new)[self.node_idx].argmax()

        return GraphState(state.x, edge_index_new, y)

    def calculate_reward(self, new_state):
        """
        Оценивает награду на основе того, изменилось ли состояние.

        :param new_state: Новое состояние графа.
        :return: Награда (число).
        """
        if not torch.equal(new_state.y, self.initial_state.y):
            return 1
        n_r = -1 / self.K
        return n_r


class ReWattPolicyNet(nn.Module):
    def __init__(self, gnn_model, penultimate_layer_embeddings_dim, mlp_hidden=16, node_idx=None, h_method='sum', pooling_method='mean', device='cpu'):
        super(ReWattPolicyNet, self).__init__()
        self.device = device

        if node_idx is not None:
            self.node_idx = node_idx
            self.graph_classification_task = False
        else:
            self.graph_classification_task = True
        self.gnn_model = gnn_model
        self.penultimate_layer_embeddings_dim = penultimate_layer_embeddings_dim

        self.edge_representer = EdgeRepresenter(h_method)
        self.graph_representer = GraphRepresenter(pooling_method)

        self.edge_fc = MLP(2 * penultimate_layer_embeddings_dim, mlp_hidden, 1).to(device)

        # Here we will not use a perceptron to determine v_fir, because in torch-geometric any graph is considered
        # directed and information about v_fir is already embedded in the edge from the adjacency list. I believe this
        # approach with a perceptron to determine v_fir was proposed by the authors of the article due to the fact that
        # they initially considered an undirected graph.
        # self.first_node_fc = MLP(3 * input_dim, mlp_hidden, 2).to(device)

        self.third_node_fc = MLP(4 * penultimate_layer_embeddings_dim, mlp_hidden, 1).to(device)

    def forward(self, state):
        embeddings = self.gnn_model.get_all_layer_embeddings(state.x, state.edge_index)[self.gnn_model.n_layers - 2]
        graph_representation = self.graph_representer(embeddings)

        E_s_t = []
        if self.graph_classification_task:
            for i in range(state.edge_index.size(1)):
                v1, v2 = state.edge_index[:, i]
                edge_representation = self.edge_representer(embeddings[v1], embeddings[v2], graph_representation)
                E_s_t.append(edge_representation)
            E_s_t = torch.stack(E_s_t)
        else:
            # edge_index_N_1 - edges that transmit information to node_idx
            _, edge_index_N_1, _, _ = k_hop_subgraph(node_idx=self.node_idx,
                                         num_hops=1,
                                         edge_index=state.edge_index,
                                         relabel_nodes=False,
                                         flow="source_to_target",
                                         directed=True)
            for i in range(edge_index_N_1.size(1)):
                v1, v2 = state.edge_index[:, i]
                edge_representation = self.edge_representer(embeddings[v1], embeddings[v2], graph_representation)
                E_s_t.append(edge_representation)
            E_s_t = torch.stack(E_s_t)

        edge_scores = self.edge_fc(E_s_t)
        edge_probs = torch.softmax(edge_scores.squeeze(), dim=0)
        edge_dist = Categorical(edge_probs)
        e_idx = edge_dist.sample()
        log_prob_edge = edge_dist.log_prob(e_idx)

        # This is an important place. I choose v_fir as state.edge_index[1][idx].item() and
        # v_sec as state.edge_index[0][idx].item() because edge (a,b) means that vertex a passes
        # information to vertex b during convolution. And we need the attacked vertex to be the one
        # that collects information. So deleting edge (a,b) changes the embedding value of vertex b.
        if self.graph_classification_task:
            v_fir = state.edge_index[1][e_idx].item()
            v_sec = state.edge_index[0][e_idx].item()
        else:
            v_fir = edge_index_N_1[1][e_idx].item()
            v_sec = edge_index_N_1[0][e_idx].item()

        # Here you can choose from which set of vertices the third v_thi will be selected.
        # The article suggests choosing from the second neighborhood. But I think that this is because they are
        # limited to only a two-layer model of the GCN for attack.
        # In the same way for both graph_classification and node_classification, we can select v_thi from all graph
        # vertices except those vertices from the first neighborhood that transmit information to the v_fir vertex
        # (parameter "source_to_target").
        N_1, _, _, _ = k_hop_subgraph(node_idx=v_fir,
                                     num_hops=1,
                                     edge_index=state.edge_index,
                                     relabel_nodes=False,
                                     flow="source_to_target",
                                     directed=True)
        N_1 = set(N_1.tolist())
        V = set([i for i in range(state.x.size(0))])
        S = list(V - N_1)

        v_et1 = torch.cat((E_s_t[e_idx], embeddings[v_fir]), dim=-1)
        V_s_t = []
        for i in range(len(S)):
            v_c = S[i]
            V_s_t.append(vc_representation(embeddings[v_c], v_et1))
        V_s_t = torch.stack(V_s_t)

        third_node_scores = self.third_node_fc(V_s_t)
        third_node_probs = torch.softmax(third_node_scores.squeeze(), dim=0)
        third_node_dist = Categorical(third_node_probs)
        v_thi_idx = third_node_dist.sample()
        log_prob_third = third_node_dist.log_prob(v_thi_idx)
        v_thi = S[v_thi_idx]

        log_prob = log_prob_edge + log_prob_third
        action = (v_fir, v_sec, v_thi)
        return action, log_prob


class ReWattAgent:
    def __init__(self, policy_net, environment, lr=1e-3, gamma=0.99):
        self.policy_net = policy_net
        self.env = environment
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Выбирает действие и log_prob.
        """
        action, log_prob = self.policy_net(state)
        return action, log_prob

    def train(self, epochs):
        stop_train = False
        for epoch in range(epochs):
            self.env.current_state = self.env.initial_state
            state = self.env.initial_state
            total_reward = 0

            for _ in range(self.env.K):
                action, log_prob = self.select_action(state)
                new_state, reward = self.env.step(action)

                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                total_reward += reward
                state = new_state

                if reward == 1:
                    stop_train = True
                    break

            self.update_policy()
            print(f"epoch {epoch}, Total Reward: {total_reward}")
            if stop_train:
                print("Class has been changed, policy net has been trained!")
                return

    def update_policy(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum([-log_prob * R for log_prob, R in zip(self.log_probs, returns)])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []


# import torch
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
#
# # Признаки вершин (например, для 3 вершин)
# x = torch.tensor([[1], [2], [3]], dtype=torch.float)  # размерность (3, 1)
#
# # edge_index описывает ребра: каждое ребро [a, b] означает, что a передает информацию b
# edge_index = torch.tensor([[0, 1, 2],  # a, b, c (исходные вершины)
#                            [1, 2, 0]], # b, c, a (целевые вершины)
#                           dtype=torch.long)
#
# # Создание объекта Data
# data = Data(x=x, edge_index=edge_index)
#
# # Инициализация слоя GCN с 1 выходным каналом
# conv = GCNConv(1, 1, bias=False, add_self_loops=False)
# conv.lin.weight.data.fill_(1)
#
# # Применение свертки
# out = conv(data.x, data.edge_index)
# print(out)
