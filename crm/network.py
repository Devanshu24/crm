from typing import Callable

import torch

from crm import Neuron


class Network:
    def __init__(self, num_neurons, adj_list, custom_activations=None):
        self.num_neurons = num_neurons
        self.adj_list = adj_list
        self.neurons = [
            Neuron(i)
            if custom_activations is None
            else Neuron(i, custom_activations[i][0], custom_activations[i][1])
            for i in range(num_neurons)
        ]
        self.weights = self._set_weights()
        self.topo_order = self._topological_sort()
        self._setup_neurons()
        self._set_output_neurons()
        self._assign_layers()
        self.has_forwarded = False
        self.is_fresh = True

    def forward(self, f_mapper):
        if not self.is_fresh:
            raise Exception(
                "Network has already been forwarded. You may want to reset it."
            )
        self.has_forwarded = True
        self.is_fresh = False
        for n_id in self.topo_order:
            if self.neurons[n_id].predeccesor_neurons:
                for pred in self.neurons[n_id].predeccesor_neurons:
                    self.neurons[n_id].value = self.neurons[n_id].value + (
                        self.weights[(pred, n_id)] * self.neurons[pred].value
                    )
                self.neurons[n_id].value = f_mapper[n_id] * self.neurons[
                    n_id
                ].activation_fn(self.neurons[n_id].value)
            else:
                self.neurons[n_id].value = torch.tensor(f_mapper[n_id])

        return torch.stack([self.neurons[i].value for i in self.output_neurons])

    def parameters(self):
        return (p for p in self.weights.values())

    def set_neuron_activation(
        self, n_id: int, activation_fn: Callable, activation_fn_grad: Callable
    ):
        self.neurons[n_id].set_activation_fn(activation_fn, activation_fn_grad)

    def reset(self):
        for n in self.neurons:
            n.value = 0
            n.grad = 0
            n.relevance = 0
        self.is_fresh = True

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
                weights[(u, v)] = torch.rand(1, requires_grad=True)
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

    def _assign_layers(self):
        """Assigns layers to neurons of the network"""
        for n in self.neurons:
            if len(n.predeccesor_neurons) == 0:
                n.layer = 0

        for n_id in self.topo_order:
            if len(self.neurons[n_id].predeccesor_neurons) > 0:
                self.neurons[n_id].layer = (
                    max(
                        [
                            self.neurons[i].layer
                            for i in self.neurons[n_id].predeccesor_neurons
                        ]
                    )
                    + 1
                )

    def lrp(self, R, n_id):
        for n in self.neurons:
            if n.relevance != 0:
                raise Exception("Relevances are not cleared, try reseting the network")
        self.neurons[n_id].relevance = R
        for n_id in self.topo_order[::-1]:
            for succ in self.neurons[n_id].successor_neurons:
                my_contribution = 1e-9
                total_contribution = 1e-9
                for pred in self.neurons[succ].predeccesor_neurons:
                    if pred == n_id:
                        my_contribution = (
                            self.neurons[n_id].value * self.weights[(pred, succ)]
                        )
                        total_contribution += my_contribution
                    else:
                        total_contribution += (
                            self.neurons[pred].value * self.weights[(pred, succ)]
                        )

                self.neurons[n_id].relevance += (
                    self.neurons[succ].relevance * my_contribution / total_contribution
                )
