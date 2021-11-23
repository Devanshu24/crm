from typing import Callable

import torch
import torch.nn.functional as F


class Neuron:
    def __init__(
        self,
        n_id: int,
        activation_fn: Callable = F.relu,
        activation_fn_grad: Callable = lambda x: 1 if x > 0 else 0,
    ):
        self.activation_fn = activation_fn
        self.activation_fn_grad = activation_fn_grad
        self.n_id = n_id
        self.value = torch.tensor(0.)
        self.grad = torch.tensor(0.)
        self.relevance = torch.tensor(0.)
        self.layer = torch.tensor(0.)
        self.predeccesor_neurons = []
        self.successor_neurons = []

    def set_successor_neurons(self, successor_neurons: list):
        self.successor_neurons = successor_neurons

    def set_predecessor_neurons(self, predeccesor_neurons: list):
        self.predeccesor_neurons = predeccesor_neurons

    def set_activation_fn(self, activation_fn: Callable, activation_fn_grad: Callable):
        self.activation_fn = activation_fn
        self.activation_fn_grad = activation_fn_grad

    def __repr__(self):
        return (
            super().__repr__()
            + f"""\n{self.n_id}: {self.value}\t Grad: {self.grad}
            \nPredecessor: {self.predeccesor_neurons}\tSuccessor: {self.successor_neurons}"""
        )

    def __str__(self):
        return f"""\n{self.n_id}: {self.value}\t Grad: {self.grad}
            \nPredecessor: {self.predeccesor_neurons}\tSuccessor: {self.successor_neurons}"""
