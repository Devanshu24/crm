from typing import Callable

import torch
import torch.nn.functional as F


class Neuron:
    def __init__(
        self,
        n_id: int,
        activation_fn: Callable = F.relu,
    ):
        self.activation_fn = activation_fn
        self.n_id = n_id
        self.value = torch.tensor(0)
        self.grad = 0
        self.relevance = 0
        self.layer = 0
        self.predecessor_neurons = []
        self.successor_neurons = []

    def set_successor_neurons(self, successor_neurons: list):
        self.successor_neurons = successor_neurons

    def set_predecessor_neurons(self, predecessor_neurons: list):
        self.predecessor_neurons = predecessor_neurons

    def set_activation_fn(self, activation_fn: Callable):
        self.activation_fn = activation_fn

    def __repr__(self):
        return (
            super().__repr__()
            + f"""\n{self.n_id}: {self.value}\t Grad: {self.grad}
            \nPredecessor: {self.predecessor_neurons}\tSuccessor: {self.successor_neurons}"""
        )

    def __str__(self):
        return f"""\n{self.n_id}: {self.value}\t Grad: {self.grad}
            \nPredecessor: {self.predecessor_neurons}\tSuccessor: {self.successor_neurons}"""
