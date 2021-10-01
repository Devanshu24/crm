from typing import Callable

import torch

from crm import Neuron


class Network:
    def __init__(self, num_neurons, adj_list):
        self.num_neurons = num_neurons
        self.adj_list = adj_list
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        self.weights = self._set_weights()
        self.topo_order = self._topological_sort()
        self._setup_neurons()
        self._set_output_neurons()

    def forward(self, f_mapper):
        for n_id in self.topo_order:
            if self.neurons[n_id].predeccesor_neurons:
                for pred in self.neurons[n_id].predeccesor_neurons:
                    self.neurons[n_id].value += (
                        self.weights[(pred, n_id)] * self.neurons[pred].value
                    )
                self.neurons[n_id].value = f_mapper[n_id] * self.neurons[
                    n_id
                ].activation_fn(self.neurons[n_id].value)
            else:
                self.neurons[n_id].value = f_mapper[n_id]

        return [self.neurons[i].value for i in self.output_neurons]

    def backward(self, f_mapper, lr, loss_val, loss_grad_fn: Callable):
        # TODO: Check backward pass
        for n_id in self.topo_order[::-1]:
            for pred in self.neurons[n_id].predeccesor_neurons:
                print(f"({n_id}, {pred})")
                if len(self.neurons[n_id].successor_neurons) == 0:

                    self.neurons[n_id].grad = loss_grad_fn(self.neurons[n_id].value)

                    self.neurons[pred].grad = (
                        loss_grad_fn(self.neurons[n_id].value)
                        * f_mapper[n_id]
                        * self.neurons[n_id].activation_fn_grad(
                            self.neurons[n_id].value
                        )
                        * self.weights[(pred, n_id)]
                    )
                    print(self.neurons[pred].grad)
                    self.weights[(pred, n_id)] = self.weights[(pred, n_id)] - (
                        lr
                        * loss_grad_fn(self.neurons[n_id].value)
                        * f_mapper[n_id]
                        * self.neurons[n_id].activation_fn_grad(
                            self.neurons[n_id].value
                        )
                        * self.neurons[pred].value.item()
                    )
                    print("HI")
                else:
                    self.neurons[pred].grad = (
                        self.neurons[n_id].grad
                        * f_mapper[n_id]
                        * self.neurons[n_id].activation_fn_grad(
                            self.neurons[n_id].value
                        )
                        * self.weights[(pred, n_id)]
                    )
                    print(self.neurons[pred].grad)
                    self.weights[(pred, n_id)] = self.weights[(pred, n_id)] - (
                        lr
                        * self.neurons[n_id].grad
                        * f_mapper[n_id]
                        * self.neurons[n_id].activation_fn_grad(
                            self.neurons[n_id].value
                        )
                        * self.neurons[pred].value
                    )

    def _set_output_neurons(self):
        self.output_neurons = []
        for n in self.neurons:
            if len(n.successor_neurons) == 0:
                self.output_neurons.append(n.n_id)

    def _setup_neurons(self):
        rev_adj_list = [[] for _ in range(self.num_neurons)]
        for i in range(self.num_neurons):
            for e in self.adj_list[i]:
                rev_adj_list[e].append(i)
        for u in self.neurons:
            u.set_successor_neurons(self.adj_list[u.n_id])
        for v in self.neurons:
            v.set_predecessor_neurons(rev_adj_list[v.n_id])

    def _set_weights(self):
        """
        This function sets the weights of the network.
        """
        weights = {}
        for u in range(self.num_neurons):
            for v in self.adj_list[u]:
                weights[(u, v)] = torch.rand(1)
        return weights

    def _topological_sort_util(self, v, visited, stack):
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.adj_list[v]:
            if visited[i] is False:
                self._topological_sort_util(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(v)

    def _topological_sort(self):
        """Returns the topological sorted order of a graph"""
        visited = [False] * self.num_neurons
        stack = []
        for i in range(self.num_neurons):
            if visited[i] is False:
                self._topological_sort_util(i, visited, stack)
        return stack[::-1]

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
