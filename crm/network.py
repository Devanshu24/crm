from crm import Neuron


class Network:
    def __init__(self, num_neurons, input_neurons, adj_list):
        self.input_neurons = input_neurons
        self.num_neurons = num_neurons
        self.adj_list = adj_list
        self.levels = {}
        self.edge_list = []  # [[u_i, v_i, weight_i]]
        self.neurons = {i: Neuron(None, None, i, None) for i in range(num_neurons)}

    def forward(self):
        for lev in range(len(self.levels)):
            neurons_in_cur_level = self.levels[lev]
            for neuron in neurons_in_cur_level:
                neuron

    def backward(self):
        pass

    def _assign_levels(self):
        """This function assigns levels to the nodes in the network, using BFS."""
        visited = set()
        cur_level = 0
        q = self.input_neurons
        self.levels[cur_level] = q
        while len(q) > 0:
            n_elem = len(q)
            while n_elem > 0:
                cur = q.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                self.levels[cur_level].append(cur)
                for neigh in self.adj_list[cur]:
                    q.append(neigh)
            cur_level += 1

    def __str__(self):
        return "\n".join(map(str, self.nodes))
