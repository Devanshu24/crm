import torch

from crm import Network


def test_network():
    assert Network(1, [[0]])


def test_topological_sort():
    n = Network(4, [[2, 3], [3], [3], []])
    assert n._topological_sort() == [1, 0, 2, 3]

    n = Network(4, [[2, 3], [3], [], []])
    assert n._topological_sort() == [1, 0, 3, 2]

    n = Network(4, [[1], [2], [3], []])
    assert n._topological_sort() == [0, 1, 2, 3]


def test_forward():
    n = Network(4, [[1], [2], [3], []])
    n.weights = {
        (0, 1): torch.tensor(1),
        (1, 2): torch.tensor(2),
        (2, 3): torch.tensor(3),
    }
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == [torch.tensor(6)]

    n = Network(6, [[3], [3, 4], [4], [5], [5], []])
    n.weights = {
        (0, 3): torch.tensor(1),
        (1, 3): torch.tensor(2),
        (1, 4): torch.tensor(3),
        (2, 4): torch.tensor(4),
        (3, 5): torch.tensor(5),
        (4, 5): torch.tensor(6),
    }
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}) == [torch.tensor(57)]
    n = Network(5, [[3], [3, 4], [4], [], []])
    n.weights = {
        (0, 3): torch.tensor(1),
        (1, 3): torch.tensor(2),
        (1, 4): torch.tensor(3),
        (2, 4): torch.tensor(4),
    }

    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}) == [
        torch.tensor(3),
        torch.tensor(7),
    ]


def test_backward():
    n = Network(4, [[1], [2], [3], []])
    n.weights = {
        (0, 1): torch.tensor(1),
        (1, 2): torch.tensor(2),
        (2, 3): torch.tensor(3),
    }
    n.forward({0: 1, 1: 1, 2: 1, 3: 1})
    n.backward({0: 1, 1: 1, 2: 1, 3: 1}, 0.01, 10, lambda x: 0.1 * x)
    assert n.neurons[3].grad == torch.tensor(0.6)
    assert n.weights[(2, 3)] == torch.tensor(3 - 0.01 * 0.6 * 2)
