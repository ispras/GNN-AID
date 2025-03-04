import random
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.data import Data
from attacks.RL_S2V.utils import norm_adj, sum_coo_tensors
import torch.nn.functional as F

# TODO add docs
class NstepReplaySubMemCell(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.actions = [None] * self.memory_size
        self.rewards = [None] * self.memory_size
        self.states = [None] * self.memory_size
        self.s_primes = [None] * self.memory_size
        self.terminals = [None] * self.memory_size

        self.count = 0
        self.current = 0

    def add(self, s_t, a_t, r_t, s_prime, terminal):
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i])

    def sample(self, batch_size):

        assert self.count >= batch_size
        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []

        for i in range(batch_size):
            idx = random.randint(0, self.count - 1)
            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(float(self.rewards[idx]))
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])

        return list_st, list_at, list_rt, list_s_primes, list_term

def hash_state_action(s_t, a_t):
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base

    key = (key * base + a_t) % base
    return key

def nipa_hash_state_action(s_t, a_t):
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base

    key = (key * base + a_t) % base
    return key

class NstepReplayMemCell(object):
    def __init__(self, memory_size, balance_sample = False):
        self.sub_list = []
        self.balance_sample = balance_sample
        self.sub_list.append(NstepReplaySubMemCell(memory_size))
        if balance_sample:
            self.sub_list.append(NstepReplaySubMemCell(memory_size))
            self.state_set = set()

    def add(self, s_t, a_t, r_t, s_prime, terminal, use_hash=True):
        if not self.balance_sample or r_t < 0:
            self.sub_list[0].add(s_t, a_t, r_t, s_prime, terminal)
        else:
            assert r_t > 0
            if use_hash:
                key = hash_state_action(s_t, a_t)
                if key in self.state_set:
                    return
                self.state_set.add(key)
            self.sub_list[1].add(s_t, a_t, r_t, s_prime, terminal)

    def sample(self, batch_size):
        if not self.balance_sample or self.sub_list[1].count < batch_size:
            return self.sub_list[0].sample(batch_size)

        list_st, list_at, list_rt, list_s_primes, list_term = self.sub_list[0].sample(batch_size // 2)
        list_st2, list_at2, list_rt2, list_s_primes2, list_term2 = self.sub_list[1].sample(batch_size - batch_size // 2)

        return list_st + list_st2, list_at + list_at2, list_rt + list_rt2, list_s_primes + list_s_primes2, list_term + list_term2

class NstepReplayMem(object):
    def __init__(self, memory_size, n_steps, balance_sample=False, model='rl_s2v'):
        self.mem_cells = []
        for i in range(n_steps - 1):
            self.mem_cells.append(NstepReplayMemCell(memory_size, False))
        self.mem_cells.append(NstepReplayMemCell(memory_size, balance_sample))

        self.n_steps = n_steps
        self.memory_size = memory_size
        self.model = model

    def add(self, s_t, a_t, r_t, s_prime, terminal, t):
        assert t >= 0 and t < self.n_steps
        if self.model == 'nipa':
            self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal, use_hash=False)
        else:
            if t == self.n_steps - 1:
                assert terminal
            else:
                assert not terminal
            self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal, use_hash=True)

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term, t):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i], t)

    def sample(self, batch_size, t = None):
        if t is None:
            t = np.random.randint(self.n_steps)
            list_st, list_at, list_rt, list_s_primes, list_term = self.mem_cells[t].sample(batch_size)
        return t, list_st, list_at, list_rt, list_s_primes, list_term

    def print_count(self):
        for i in range(self.n_steps):
            for j, cell in enumerate(self.mem_cells[i].sub_list):
                print('Cell {} sub_list {}: {}'.format(i, j, cell.count))


class QNetNode(nn.Module):

    def __init__(self, num_node_features, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        '''
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        '''
        super(QNetNode, self).__init__()
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        self.num_node_features = num_node_features

        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm

        if bilin_q:
            last_wout = embed_dim
        else:
            last_wout = 1
            self.bias_target = Parameter(torch.Tensor(1, embed_dim))

        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)

        self.w_n2l = Parameter(torch.Tensor(num_node_features, embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        # self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)

        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp.cuda()
        return sp

    def forward(self, node_features, orig_edge_index, orig_edge_weight, num_nodes, time_t, states, actions, greedy_acts=False, is_inference=False):

        if node_features.data.is_sparse:
            input_node_linear = torch.spmm(node_features, self.w_n2l)
        else:
            input_node_linear = torch.mm(node_features, self.w_n2l)

        input_node_linear += self.bias_n2l

        target_nodes, batch_graph, picked_nodes = zip(*states)

        list_pred = []
        prefix_sum = []
        for i in range(len(batch_graph)):
            region = self.list_action_space[target_nodes[i]]

            node_embed = input_node_linear.clone()
            if picked_nodes is not None and picked_nodes[i] is not None:
                with torch.set_grad_enabled(mode=not is_inference):
                    picked_sp =  self.make_spmat(self.total_nodes, 1, picked_nodes[i], 0)
                    node_embed += torch.spmm(picked_sp, self.bias_picked)
                    region = self.list_action_space[picked_nodes[i]]

            if not self.bilin_q:
                with torch.set_grad_enabled(mode=not is_inference):
                # with torch.no_grad():
                    target_sp = self.make_spmat(self.total_nodes, 1, target_nodes[i], 0)
                    node_embed += torch.spmm(target_sp, self.bias_target)

            with torch.set_grad_enabled(mode=not is_inference):
                device = node_features.device
                extra_edge_index, extra_edge_weight = batch_graph[i].get_extra_adj(device=device)
                if extra_edge_index is None:
                    normed_edge_ind, normed_edge_weight = norm_adj(orig_edge_index, orig_edge_weight, num_nodes)
                    adj = torch.sparse.FloatTensor(normed_edge_ind, normed_edge_weight, torch.Size([num_nodes, num_nodes]))
                else:
                    modified_edge_index, modified_edge_weight = sum_coo_tensors(orig_edge_index, orig_edge_weight,
                                                                                extra_edge_index, extra_edge_weight,
                                                                                num_nodes)
                    modified_edge_index, modified_edge_weight = norm_adj(modified_edge_index, modified_edge_weight,
                                                                         gm=self.gm)
                    adj = torch.sparse.FloatTensor(modified_edge_index, modified_edge_weight, torch.Size([num_nodes, num_nodes]))
                # adj = self.norm_tool.norm_extra(batch_graph[i].get_extra_adj(device, return_sparse=True))


                lv = 0
                input_message = node_embed

                node_embed = F.relu(input_message)
                while lv < self.max_lv:
                    n2npool = torch.spmm(adj, node_embed)
                    node_linear = self.conv_params( n2npool )
                    merged_linear = node_linear + input_message
                    node_embed = F.relu(merged_linear)
                    lv += 1

                target_embed = node_embed[target_nodes[i], :].view(-1, 1)
                if region is not None:
                    node_embed = node_embed[region]

                graph_embed = torch.mean(node_embed, dim=0, keepdim=True)

                if actions is None:
                    graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
                else:
                    if region is not None:
                        act_idx = region.index(actions[i])
                    else:
                        act_idx = actions[i]
                    node_embed = node_embed[act_idx, :].view(1, -1)

                embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
                if self.mlp_hidden:
                    embed_s_a = F.relu( self.linear_1(embed_s_a) )
                raw_pred = self.linear_out(embed_s_a)

                if self.bilin_q:
                    raw_pred = torch.mm(raw_pred, target_embed)
                list_pred.append(raw_pred)

        if greedy_acts:
            actions, _ = node_greedy_actions(target_nodes, picked_nodes, list_pred, self)

        return actions, list_pred

class NStepQNetNode(nn.Module):

    def __init__(self, node_features, num_steps, num_node_features, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):

        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        list_mod = []
        for i in range(0, num_steps):
            # list_mod.append(QNetNode(node_features, node_labels, list_action_space))
            list_mod.append(QNetNode(num_node_features, list_action_space, bilin_q, embed_dim, mlp_hidden, max_lv, gm=gm, device=device))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, orig_edge_index, orig_edge_weigt, num_nodes, time_t, states, actions, greedy_acts = False, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](self.node_features, orig_edge_index, orig_edge_weigt, num_nodes, time_t, states, actions, greedy_acts, is_inference)

def node_greedy_actions(target_nodes, picked_nodes, list_q, net):
    assert len(target_nodes) == len(list_q)

    actions = []
    values = []
    for i in range(len(target_nodes)):
        region = net.list_action_space[target_nodes[i]]
        if picked_nodes is not None and picked_nodes[i] is not None:
            region = net.list_action_space[picked_nodes[i]]
        if region is None:
            assert list_q[i].size()[0] == net.total_nodes
        else:
            assert len(region) == list_q[i].size()[0]

        val, act = torch.max(list_q[i], dim=0)
        values.append(val)
        if region is not None:
            act = region[act.data.cpu().numpy()[0]]
            # act = Variable(torch.LongTensor([act]))
            act = torch.LongTensor([act])
            actions.append(act)
        else:
            actions.append(act)

    return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)