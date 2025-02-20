import os
import os.path as osp
import sys
import numpy as np
import torch
import networkx as nx
import random

from torch_geometric.data.remote_backend_utils import num_nodes

import utils
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import List, Dict
from copy import deepcopy

from attacks.RL_S2V.utils import sum_coo_tensors, norm_adj
from base.datasets_processing import GeneralDataset
from q_learning import NstepReplayMem, NStepQNetNode, node_greedy_actions


# class StaticGraph(object):
#     graph = None
#
#     @staticmethod
#     def get_gsize():
#         return torch.Size( (len(StaticGraph.graph), len(StaticGraph.graph)) )
#

# class GraphNormTool(object):
#
#     def __init__(self, gm, device):
#         self.gm = gm
#         edges = np.array(g.edges(), dtype=np.int64)
#         rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
#
#         # self_edges = np.array([range(len(g)), range(len(g))], dtype=np.int64)
#         # edges = np.hstack((edges.T, rev_edges, self_edges))
#         edges = np.hstack((edges.T, rev_edges))
#         idxes = torch.LongTensor(edges)
#         values = torch.ones(idxes.size()[1])
#
#         self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
#         self.raw_adj = self.raw_adj.to(device)
#
#         self.normed_adj = self.raw_adj.clone()
#         if self.gm == 'gcn':
#             self.normed_adj = utils.normalize_sparse_tensor(self.normed_adj, sparse=True)
#             # GraphLaplacianNorm(self.normed_adj)
#         else:
#
#             self.normed_adj = utils.degree_normalize_sparse_tensor(self.normed_adj, sparse=True)
#             # GraphDegreeNorm(self.normed_adj)
#
#     def norm_extra(self, added_adj = None):
#         if added_adj is None:
#             return self.normed_adj
#
#         new_adj = self.raw_adj + added_adj
#         if self.adj_norm:
#             if self.gm == 'gcn':
#                 new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
#             else:
#                 new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse=True)
#
#         return new_adj


class ModifiedGraph(object):

    def __init__(
            self,
            num_nodes,
            directed_edges = None,
            weights = None
    ):
        self.edge_set = set()  #(first, second)
        self.num_nodes = num_nodes
        self.node_set = np.arange(self.num_nodes)
        if directed_edges is not None:
            self.directed_edges = deepcopy(directed_edges)
            self.weights = deepcopy(weights)
        else:
            self.directed_edges = []
            self.weights = []

    def add_edge(self, x, y, z):
        assert x is not None and y is not None
        if x == y:
            return
        for e in self.directed_edges:
            if e[0] == x and e[1] == y:
                return
            if e[1] == x and e[0] == y:
                return
        self.edge_set.add((x, y)) # (first, second)
        self.edge_set.add((y, x)) # (second, first)
        self.directed_edges.append((x, y))
        # assert z < 0
        self.weights.append(z)

    def get_extra_adj(self, device, return_sparse=False):
        if len(self.directed_edges):
            edges = np.array(self.directed_edges, dtype=np.int64)
            rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
            edges = np.hstack((edges.T, rev_edges))

            idxes = torch.LongTensor(edges)
            values = torch.Tensor(self.weights + self.weights)
            if return_sparse:
                added_adj = torch.sparse.FloatTensor(idxes, values, self.num_nodes)
                added_adj = added_adj.to(device)
                return added_adj
            else:
                idxes.to(device)
                values.to(device)
                return idxes, values
        else:
            return None

    def get_possible_nodes(self, target_node):
        # connected = set()
        connected = [target_node]
        for n1, n2 in self.edge_set:
            if n1 == target_node:
                # connected.add(target_node)
                connected.append(n2)
        return np.setdiff1d(self.node_set, np.array(connected))

        # return self.node_set - connected

class NodeAttackEnv(object):
    """Node attack environment. It executes an action and then change the
    environment status (modify the graph).
    """

    def __init__(
            self,
            gen_dataset: GeneralDataset,
            all_targets: List,
            list_action_space: Dict[List],
            classifier: torch.nn.Module,
            num_mod: int = 1,
            reward_type: str = 'binary'
    ) -> None:

        self.classifier = classifier
        self.list_action_space = list_action_space
        self.all_targets = all_targets
        self.num_mod = num_mod
        self.reward_type = reward_type
        self.gen_dataset = gen_dataset
        self.num_nodes = gen_dataset.dataset.data.x.shape[0]

    def setup(
            self,
            target_nodes: List
    ) -> None:
        self.target_nodes = target_nodes
        self.n_steps = 0
        self.first_nodes = None
        self.rewards = None
        self.binary_rewards = None
        self.modified_list = []
        for i in range(len(self.target_nodes)):
            self.modified_list.append(ModifiedGraph())

        self.list_acc_of_all = []

    def step(
            self,
            actions: List
    ) -> None:
        """run actions and get rewards
        """
        data = self.gen_dataset.dataset.data
        if self.first_nodes is None: # pick the first node of edge
            assert self.n_steps % 2 == 0
            self.first_nodes = actions[:]
        else:
            for i in range(len(self.target_nodes)):
                # assert self.first_nodes[i] != actions[i]
                # deleta an edge from the graph
                self.modified_list[i].add_edge(self.first_nodes[i], actions[i], -1.0)
            self.first_nodes = None
            self.banned_list = None
        self.n_steps += 1

        if self.isTerminal():
            # only calc reward when its terminal
            acc_list = []
            loss_list = []
            # for i in tqdm(range(len(self.target_nodes))):
            for i in (range(len(self.target_nodes))):
                device = self.labels.device
                extra_edge_index, extra_edge_weight = self.modified_list[i].get_extra_adj(device=device)
                modified_edge_index, modified_edge_weight = sum_coo_tensors(data.edge_index, data.edge_weight,
                                                                            extra_edge_index, extra_edge_weight,
                                                                            self.num_nodes)
                modified_edge_index, modified_edge_weight = norm_adj(modified_edge_index, modified_edge_weight)
                #adj = self.classifier.norm_tool.norm_extra(extra_adj)

                output = self.classifier(data.x, data.edge_index, data.edge_weight)

                loss, acc = loss_acc(output, self.labels, self.all_targets, avg_loss=False)
                # _, loss, acc = self.classifier(self.features, Variable(adj), self.all_targets, self.labels, avg_loss=False)

                cur_idx = self.all_targets.index(self.target_nodes[i])
                acc = np.copy(acc.double().cpu().view(-1).numpy())
                loss = loss.data.cpu().view(-1).numpy()
                self.list_acc_of_all.append(acc)
                acc_list.append(acc[cur_idx])
                loss_list.append(loss[cur_idx])

            self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            if self.reward_type == 'binary':
                self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            else:
                assert self.reward_type == 'nll'
                self.rewards = np.array(loss_list).astype(np.float32)

    def sample_pos_rewards(self, num_samples):
        assert self.list_acc_of_all is not None
        cands = []

        for i in range(len(self.list_acc_of_all)):
            succ = np.where( self.list_acc_of_all[i] < 0.9 )[0]

            for j in range(len(succ)):

                cands.append((i, self.all_targets[succ[j]]))

        if num_samples > len(cands):
            return cands
        random.shuffle(cands)
        return cands[0:num_samples]

    def uniformRandActions(self) -> List:
        # TODO: here only support deleting edges
        # seems they sample first node from 2-hop neighbours
        act_list = []
        for i in range(len(self.target_nodes)):
            cur_node = self.target_nodes[i]
            region = self.list_action_space[cur_node]

            if self.first_nodes is not None and self.first_nodes[i] is not None:
                region = self.list_action_space[self.first_nodes[i]]

            if region is None:  # singleton node
                cur_action = np.random.randint(len(self.list_action_space))
            else: # select from neighbours or 2-hop neighbours
                cur_action = region[np.random.randint(len(region))]

            act_list.append(cur_action)
        return act_list

    def isTerminal(self) -> bool:
        if self.n_steps == 2 * self.num_mod:
            return True
        return False

    def getStateRef(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes

        return zip(self.target_nodes, self.modified_list, cp_first)

    def cloneState(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]

        return list(zip(self.target_nodes[:], deepcopy(self.modified_list), cp_first))


class RLS2VAgent(object):
    """ Reinforcement learning agent for RL-S2V attack.

    Parameters
    ----------
    env :
        Node attack environment
    features :
        node features matrix
    labels :
        labels
    idx_meta :
        node meta indices
    idx_test :
        node test indices
    list_action_space : list
        list of action space
    num_mod :
        number of modification (perturbation) on the graph
    reward_type : str
        type of reward (e.g., 'binary')
    batch_size :
        batch size for training DQN
    save_dir :
        saving directory for model checkpoints
    device: str
        'cpu' or 'cuda'
    """

    def __init__(
            self,
            env: NodeAttackEnv,
            gen_dataset: GeneralDataset,
            idx_meta,
            idx_test,
            list_action_space,
            num_mod,
            reward_type,
            batch_size=10,
            num_wrong=0,
            bilin_q=1,
            embed_dim=64,
            gm='mean_field',
            mlp_hidden=64,
            max_lv=1,
            save_dir='checkpoint_dqn',
            device=None
    ):

        assert device is not None, "'device' cannot be None, please specify it"

        self.idx_meta = idx_meta
        self.idx_test = idx_test
        self.num_wrong = num_wrong
        self.list_action_space = list_action_space
        self.num_mod = num_mod
        self.reward_type = reward_type
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_node_features = gen_dataset.dataset.data.x.shape[1]
        if not osp.exists(save_dir):
            os.system('mkdir -p {}'.format(save_dir))

        self.gm = gm
        self.device = device

        self.mem_pool = NstepReplayMem(memory_size=500000, n_steps=2 * num_mod, balance_sample=reward_type == 'binary')
        self.env = env

        # self.net = QNetNode(features, labels, list_action_space)
        # self.old_net = QNetNode(features, labels, list_action_space)
        self.net = NStepQNetNode(2 * num_mod, self.num_node_features, list_action_space,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.old_net = NStepQNetNode(2 * num_mod, self.num_node_features, list_action_space,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.net = self.net.to(device)
        self.old_net = self.old_net.to(device)

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = 10
        self.step = 0
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

        self.trained = False

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:
            cur_state = self.env.getStateRef()
            actions, values = self.net(time_t, cur_state, None, greedy_acts=True, is_inference=True)
            actions = list(actions.cpu().numpy())

        return actions

    def run_simulation(self):

        if (self.pos + 1) * self.batch_size > len(self.idx_test):
            self.pos = 0
            random.shuffle(self.idx_test)

        selected_idx = self.idx_test[self.pos * self.batch_size : (self.pos + 1) * self.batch_size]
        self.pos += 1
        self.env.setup(selected_idx)

        t = 0
        list_of_list_st = []
        list_of_list_at = []

        while not self.env.isTerminal():
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()

            self.env.step(list_at)

            # TODO Wei added line #87
            env = self.env
            assert (env.rewards is not None) == env.isTerminal()
            if env.isTerminal():
                rewards = env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t)
            list_of_list_st.append( deepcopy(list_st) )
            list_of_list_at.append( deepcopy(list_at) )
            t += 1

        # if the reward type is nll_loss, directly return
        if self.reward_type == 'nll':
            return

        T = t
        cands = self.env.sample_pos_rewards(len(selected_idx))
        if len(cands):
            for c in cands:
                sample_idx, target = c
                doable = True
                for t in range(T):
                    if self.list_action_space[target] is not None and (not list_of_list_at[t][sample_idx] in self.list_action_space[target]):
                        doable = False # TODO WHY False? This is only 1-hop neighbour
                        break
                if not doable:
                    continue

                for t in range(T):
                    s_t = list_of_list_st[t][sample_idx]
                    a_t = list_of_list_at[t][sample_idx]
                    s_t = [target, deepcopy(s_t[1]), s_t[2]]
                    if t + 1 == T:
                        s_prime = (None, None, None)
                        r = 1.0
                        term = True
                    else:
                        s_prime = list_of_list_st[t + 1][sample_idx]
                        s_prime = [target, deepcopy(s_prime[1]), s_prime[2]]
                        r = 0.0
                        term = False
                    self.mem_pool.mem_cells[t].add(s_t, a_t, r, s_prime, term)

    def eval(self, training=True):
        """Evaluate RL agent.
        """

        self.env.setup(self.idx_meta)
        t = 0

        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1

        acc = 1 - (self.env.binary_rewards + 1.0) / 2.0
        acc = np.sum(acc) / (len(self.idx_meta) + self.num_wrong)
        print('\033[93m average test: acc %.5f\033[0m' % (acc))

        if training == True and self.best_eval is None or acc < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), osp.join(self.save_dir, 'epoch-best.model'))
            with open(osp.join(self.save_dir, 'epoch-best.txt'), 'w') as f:
                f.write('%.4f\n' % acc)
            with open(osp.join(self.save_dir, 'attack_solution.txt'), 'w') as f:
                for i in range(len(self.idx_meta)):
                    f.write('%d: [' % self.idx_meta[i])
                    for e in self.env.modified_list[i].directed_edges:
                        f.write('(%d %d)' % e)
                    f.write('] succ: %d\n' % (self.env.binary_rewards[i]))
            self.best_eval = acc

    def train(self, num_steps=100000, lr=0.001):
        """Train RL agent.
        """

        pbar = tqdm(range(self.burn_in), unit='batch')

        for p in pbar:
            self.run_simulation()

        pbar = tqdm(range(num_steps), unit='steps')
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        for self.step in pbar:

            self.run_simulation()

            if self.step % 123 == 0:
                # update the params of old_net
                self.take_snapshot()
            if self.step % 500 == 0:
                self.eval()

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt).to(self.device)

            if not list_term[0]:
                target_nodes, _, picked_nodes = zip(*list_s_primes)
                _, q_t_plus_1 = self.old_net(cur_time + 1, list_s_primes, None)
                _, q_rhs = node_greedy_actions(target_nodes, picked_nodes, q_t_plus_1, self.old_net)
                list_target += q_rhs

            # list_target = Variable(list_target.view(-1, 1))
            list_target = list_target.view(-1, 1)
            _, q_sa = self.net(cur_time, list_st, list_at)
            q_sa = torch.cat(q_sa, dim=0)
            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )

def loss_acc(output, labels, targets, avg_loss=True):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()[targets]
    loss = F.nll_loss(output[targets], labels[targets], reduction='mean' if avg_loss else 'none')

    if avg_loss:
        return loss, correct.sum() / len(targets)
    return loss, correct