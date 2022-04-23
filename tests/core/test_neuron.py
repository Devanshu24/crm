import torch

from crm.core import Neuron


def test_neuron():
    Neuron(0)
    Neuron(0, lambda x: 10 * x)


def test_successor():
    n = Neuron(0)
    successors = [Neuron(1), Neuron(2), Neuron(3)]
    n.set_successor_neurons(successors)
    assert n.successor_neurons == successors


def test_predecessor():
    n = Neuron(0)
    predecessors = [Neuron(1), Neuron(2), Neuron(3)]
    n.set_predecessor_neurons(predecessors)
    assert n.predecessor_neurons == predecessors


def test_activation():
    n = Neuron(0)
    assert n.activation_fn(torch.tensor(-1)) == torch.tensor(0)
    assert n.activation_fn(torch.tensor(10)) == torch.tensor(10)
    n = Neuron(0, lambda x: 10 * x)
    assert n.activation_fn(torch.tensor(1)) == torch.tensor(10)


# def test_activation_fn_grad():
#     n = Neuron(0)
#     assert n.activation_fn_grad(torch.tensor(-1)) == torch.tensor(0)
#     assert n.activation_fn_grad(torch.tensor(10)) == torch.tensor(1)
#     n = Neuron(0, lambda x: 10 * x)
#     assert n.activation_fn_grad(torch.tensor(12)) == torch.tensor(10)
