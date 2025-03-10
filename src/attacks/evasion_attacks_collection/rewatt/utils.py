import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.utils import k_hop_subgraph
from models_builder.gnn_constructor import FrameworkGNNConstructor
from tqdm import tqdm
from typing import Callable, Tuple


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class EdgeRepresenter:
    def __init__(
            self,
            method: str = 'sum'
    ):
        super(EdgeRepresenter, self).__init__()
        self.h = self._get_h_function(method)   # h() function from article

    def __call__(
            self,
            v1_emb: torch.Tensor,
            v2_emb: torch.Tensor,
            graph_emb: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((graph_emb, self.h(v1_emb, v2_emb)), dim=-1)

    @staticmethod
    def _get_h_function(
            method: str,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if method == "sum":
            return lambda a, b: a + b
        elif method == "mul":
            return lambda a, b: a * b
        elif method == "max":
            return lambda a, b: torch.where(a.norm(dim=-1) > b.norm(dim=-1), a, b)
        else:
            raise ValueError("Unsupported method")


class GraphRepresenter:
    def __init__(
            self,
            method: str = 'mean'
    ):
        super(GraphRepresenter, self).__init__()
        self.method = method

    def __call__(
            self,
            node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.method == 'mean':
            return torch.mean(node_embeddings, dim=0)
        elif self.method == 'max':
            return torch.max(node_embeddings, dim=0)[0]
        else:
            raise ValueError("Unsupported pooling method")


def vc_representation(
        vc_emb: torch.Tensor,
        v_et1: torch.Tensor
) -> torch.Tensor:
    """
    Parameters:
    - v1_emb: The embedding of v_fir node.
    - edge_repr: Representation of an edge in the context of nodes v_fir and v_sec.
    - vc_emb: The embedding of v_c node.
    """
    return torch.cat((v_et1, vc_emb), dim=-1)


class GraphState:
    def __init__(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            y: torch.Tensor,
            y_prob: float
    ):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.y_prob = y_prob


class Action:
    def __init__(
            self,
            v_fir: int,
            v_sec: int,
            v_thi: int
    ):
        self.v_fir = v_fir
        self.v_sec = v_sec
        self.v_thi = v_thi


class GraphEnvironment:
    def __init__(
            self,
            gnn_model: FrameworkGNNConstructor,
            initial_state: GraphState,
            eps: float,
            node_idx: int = None
    ):
        """ An environment class that receives agent actions and updates the state. """
        self.gnn_model = gnn_model
        self.K = int(eps * initial_state.edge_index.size(1))
        self.initial_state = initial_state
        self.current_state = initial_state
        if node_idx is not None:
            self.node_idx = node_idx
            self.graph_classification_task = False
        else:
            self.graph_classification_task = True

    def step(
            self,
            action: Action
    ) -> Tuple[GraphState, float]:
        """
        Performs an action, changes the state, and returns the new state and reward.

        :param action: Action.
        :return: (new state, reward).
        """
        new_state = self.apply_rewiring(self.current_state, action)
        reward = self.calculate_reward(new_state)
        self.current_state = new_state
        return new_state, reward

    def apply_rewiring(
            self,
            state: GraphState,
            action: Action
    ) -> GraphState:
        """
        Applies an edge rewiring operation to the current state.

        :param state: current state of the graph.
        :param action: Action - nodes involved in rewiring.
        :return: new state of the graph (GraphState).
        """
        v_fir, v_sec, v_thi = action.v_fir, action.v_sec, action.v_thi

        # copy the edges so as not to change the original data
        edge_index_new = state.edge_index.clone()

        # looking for the edge that needs to be removed
        mask = (edge_index_new[0] == v_sec) & (edge_index_new[1] == v_fir)

        # remove edge (v_sec, v_fir) and add (v_thi, v_fir) !!!
        edge_index_new = torch.cat([
            edge_index_new[:, ~mask],  # leave only those edges that are not deleted
            torch.tensor([[v_thi], [v_fir]], dtype=torch.long)  # add a new edge
        ], dim=1)

        if self.graph_classification_task:
            probs = torch.softmax(self.gnn_model(state.x, edge_index_new), dim=1).squeeze()
        else:
            probs = torch.softmax(self.gnn_model(state.x, edge_index_new)[self.node_idx], dim=0)
        y = probs.argmax()
        y_prob = probs.max().item()

        return GraphState(state.x, edge_index_new, y, y_prob)

    def calculate_reward(
            self,
            new_state: GraphState
    ) -> float:
        if not torch.equal(new_state.y, self.initial_state.y):
            return 1
        n_r = -1 / self.K
        return n_r
        # return self.initial_state.y_prob - new_state.y_prob


class ReWattPolicyNet(nn.Module):
    def __init__(
            self,
            gnn_model: FrameworkGNNConstructor,
            penultimate_layer_embeddings_dim: int,
            mlp_hidden: int = 16,
            node_idx: int = None,
            h_method: str = 'sum',
            pooling_method: str = 'mean',
            device: str = 'cpu'
    ):
        super(ReWattPolicyNet, self).__init__()
        self.device = device

        if node_idx is not None:
            self.node_idx = node_idx
            self.graph_classification_task = False
        else:
            self.graph_classification_task = True

        self.gnn_model = gnn_model
        # Disable updating of model parameters
        for param in self.gnn_model.parameters():
            param.requires_grad = False

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

    def forward(
            self,
            state: GraphState
    ) -> Tuple[Action, float]:
        embeddings = self.gnn_model.get_all_layer_embeddings(state.x, state.edge_index)[self.gnn_model.n_layers - 2]
        graph_representation = self.graph_representer(embeddings)

        if self.graph_classification_task:
            v_fir_candidate_edges = state.edge_index
        else:
            # "source_to_target" - edges that transmit information to node_idx: (v_i, node_idx) v_i -> node_idx
            _, v_fir_candidate_edges, _, _ = k_hop_subgraph(node_idx=self.node_idx,
                                                            num_hops=1,
                                                             edge_index=state.edge_index,
                                                             relabel_nodes=False,
                                                             flow="source_to_target",
                                                             directed=True)

        E_s_t = []
        for i in range(v_fir_candidate_edges.size(1)):
            v1, v2 = v_fir_candidate_edges[:, i]
            edge_representation = self.edge_representer(embeddings[v1], embeddings[v2], graph_representation)
            E_s_t.append(edge_representation)
        E_s_t = torch.stack(E_s_t)

        edge_scores = self.edge_fc(E_s_t)
        edge_probs = torch.softmax(edge_scores.squeeze(), dim=0)
        edge_dist = Categorical(edge_probs)
        e_idx = edge_dist.sample()
        log_prob_edge = edge_dist.log_prob(e_idx)

        # This is an important place. I choose v_fir as v_fir_candidate_edges[1][idx].item() and
        # v_sec as v_fir_candidate_edges[0][idx].item() because edge (a,b) means that vertex a passes
        # information to vertex b during convolution. And we need the attacked vertex to be the one
        # that collects information. So deleting edge (a,b) changes the embedding value of vertex b.
        # So in .apply_rewiring() method we delete (v_sec, v_fir) edge
        v_fir = v_fir_candidate_edges[1][e_idx].item()
        v_sec = v_fir_candidate_edges[0][e_idx].item()

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
        action = Action(v_fir, v_sec, v_thi)
        return action, log_prob


class ReWattAgent:
    def __init__(
            self,
            policy_net: ReWattPolicyNet,
            environment: GraphEnvironment,
            lr: float = 1e-3,
            gamma: float = 0.99
    ):
        self.policy_net = policy_net
        self.env = environment
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(
            self,
            state: GraphState
    ) -> Tuple[Action, float]:
        action, log_prob = self.policy_net(state)
        return action, log_prob

    def train(
            self,
            epochs: int
    ) -> GraphState:
        best_y_prob = self.env.initial_state.y_prob
        best_state = GraphState
        for epoch in tqdm(range(epochs)):
            self.env.current_state = self.env.initial_state
            state = self.env.initial_state
            total_reward = 0

            new_state = GraphState
            for _ in range(self.env.K):
                action, log_prob = self.select_action(state)
                new_state, reward = self.env.step(action)

                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                total_reward += reward
                state = new_state

                if reward == 1:
                     self.update_policy()
                     print("Class has been changed, policy net has been trained!")
                     return new_state

            self.update_policy()

            # look at y_prob after K rewirings and save state of the graph that most strongly reduces
            # the probability of a correct prediction
            if new_state.y_prob < best_y_prob:
                best_y_prob = new_state.y_prob
                best_state = new_state
        return best_state

    def update_policy(
            self
    ) -> None:
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
