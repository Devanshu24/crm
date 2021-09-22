from typing import Callable


class Neuron:
    def __init__(
        self, predicate_fn: Callable, activation_fn: Callable, n_id: int, layer: int
    ):
        self.predicate_fn = predicate_fn
        self.activation_fn = activation_fn
        self.n_id = n_id
        self.layer = layer
        self.value = None
        self.weights = {}
        self.predeccesor_neurons = []
        self.successor_neurons = []

    def __str__(self):
        return f"{self.id}: Predicate Fn: {self.predicate_fn.__doc__}\tActivation Fn: {self.activation_fn.__doc__}"
