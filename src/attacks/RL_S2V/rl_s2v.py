from attacks.evasion_attacks import EvasionAttacker


class RLS2VAttack(EvasionAttacker):
    name = "RLS2VAttacker"

    def __init__(self):
        super().__init__()

    def attack(self, gen_dataset):
        pass


class NodeAttackEnv(object):
    """Node attack environment. It executes an action and then change the
    environment status (modify the graph).
    """

    def __init__(self, gen_dataset, all_targets, list_action_space, classifier, num_mod=1, reward_type='binary'):
        self.dataset = gen_dataset
        self.classifier = classifier
        self.list_action_space = list_action_space
        self.all_targets = all_targets
        self.num_mod = num_mod
        self.reward_type = reward_type

    def setup(self, target_nodes):
        self.target_nodes = target_nodes
        self.n_steps = 0
        self.first_nodes = None
        self.rewards = None
        self.binary_rewards = None
        self.modified_list = []
        for i in range(len(self.target_nodes)):
            self.modified_list.append(ModifiedGraph())

        self.list_acc_of_all = []

    def step(self, actions):
        """run actions and get rewards
        """
        if self.first_nodes is None:  # pick the first node of edge
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
                extra_adj = self.modified_list[i].get_extra_adj(device=device)
                adj = self.classifier.norm_tool.norm_extra(extra_adj)

                output = self.classifier(self.features, adj)

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
            succ = np.where(self.list_acc_of_all[i] < 0.9)[0]

            for j in range(len(succ)):
                cands.append((i, self.all_targets[succ[j]]))

        if num_samples > len(cands):
            return cands
        random.shuffle(cands)
        return cands[0:num_samples]

    def uniformRandActions(self):
        # TODO: here only support deleting edges
        # seems they sample first node from 2-hop neighbours
        act_list = []
        for i, n in enumerate(self.target_nodes):
            region = self.list_action_space[n]

            if self.first_nodes is not None and self.first_nodes[i] is not None:
                region = self.list_action_space[self.first_nodes[i]]

            if region is None:  # singleton node
                cur_action = np.random.randint(len(self.list_action_space))
            else:  # select from neighbours or 2-hop neighbours
                cur_action = region[np.random.randint(len(region))]

            act_list.append(cur_action)
        return act_list

    def isTerminal(self):
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
