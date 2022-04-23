import pytest
import torch

from crm.core import Network


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
        (0, 1): torch.tensor(1.0, requires_grad=True),
        (1, 2): torch.tensor(2.0, requires_grad=True),
        (2, 3): torch.tensor(3.0, requires_grad=True),
    }
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == torch.tensor(6.0).reshape(1, 1)

    n = Network(6, [[3], [3, 4], [4], [5], [5], []])
    n.weights = {
        (0, 3): torch.tensor(1.0, requires_grad=True),
        (1, 3): torch.tensor(2.0, requires_grad=True),
        (1, 4): torch.tensor(3.0, requires_grad=True),
        (2, 4): torch.tensor(4.0, requires_grad=True),
        (3, 5): torch.tensor(5.0, requires_grad=True),
        (4, 5): torch.tensor(6.0, requires_grad=True),
    }
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}) == torch.tensor(
        57.0
    ).reshape(1, 1)
    n = Network(5, [[3], [3, 4], [4], [], []])
    n.weights = {
        (0, 3): torch.tensor(1.0, requires_grad=True),
        (1, 3): torch.tensor(2.0, requires_grad=True),
        (1, 4): torch.tensor(3.0, requires_grad=True),
        (2, 4): torch.tensor(4.0, requires_grad=True),
    }

    assert torch.allclose(
        n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
        torch.stack([torch.tensor(3.0), torch.tensor(7.0)]),
    )


def test_fast_forward():
    n = Network(4, [[1], [2], [3], []])
    n.weights = {
        (0, 1): torch.tensor(1.0, requires_grad=True),
        (1, 2): torch.tensor(2.0, requires_grad=True),
        (2, 3): torch.tensor(3.0, requires_grad=True),
    }
    assert n.fast_forward({0: 1, 1: 1, 2: 1, 3: 1}) == torch.tensor(6.0).reshape(1, 1)

    n = Network(6, [[3], [3, 4], [4], [5], [5], []])
    n.weights = {
        (0, 3): torch.tensor(1.0, requires_grad=True),
        (1, 3): torch.tensor(2.0, requires_grad=True),
        (1, 4): torch.tensor(3.0, requires_grad=True),
        (2, 4): torch.tensor(4.0, requires_grad=True),
        (3, 5): torch.tensor(5.0, requires_grad=True),
        (4, 5): torch.tensor(6.0, requires_grad=True),
    }
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}) == torch.tensor(
        57.0
    ).reshape(1, 1)
    n = Network(5, [[3], [3, 4], [4], [], []])
    n.weights = {
        (0, 3): torch.tensor(1.0, requires_grad=True),
        (1, 3): torch.tensor(2.0, requires_grad=True),
        (1, 4): torch.tensor(3.0, requires_grad=True),
        (2, 4): torch.tensor(4.0, requires_grad=True),
    }

    assert torch.allclose(
        n.forward({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}),
        torch.stack([torch.tensor(3.0), torch.tensor(7.0)]),
    )


def test_no_double_forward():
    n = Network(4, [[1], [2], [3], []])
    n.forward({0: 1, 1: 1, 2: 1, 3: 1})
    with pytest.raises(Exception):
        n.forward({0: 1, 1: 1, 2: 1, 3: 1})


def test_set_neuron_activation():
    n = Network(4, [[1], [2], [3], []])
    n.weights = {
        (0, 1): torch.tensor(1.0, requires_grad=True),
        (1, 2): torch.tensor(2.0, requires_grad=True),
        (2, 3): torch.tensor(3.0, requires_grad=True),
    }

    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == torch.tensor(6.0).reshape(1, 1)
    n.reset()
    n.set_neuron_activation(
        3,
        lambda x: torch.exp(x.float()),
    )
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == torch.exp(torch.tensor(6.0)).reshape(
        1, 1
    )


def test_assign_layers():
    n = Network(4, [[2], [2, 3], [3], []])
    assert n.neurons[0].layer == 0
    assert n.neurons[1].layer == 0
    assert n.neurons[2].layer == 1
    assert n.neurons[3].layer == 2


def test_lrp():
    n = Network(4, [[2], [2, 3], [3], []])
    n.weights = {
        (0, 2): torch.tensor(1.0, requires_grad=True),
        (1, 2): torch.tensor(3.0, requires_grad=True),
        (1, 3): torch.tensor(2.0, requires_grad=True),
        (2, 3): torch.tensor(2.0, requires_grad=True),
    }
    n.forward({0: -3, 1: 2, 2: 1, 3: 1})
    assert [n.neurons[i].value for i in range(len(n.neurons))] == [
        torch.tensor(-3),
        torch.tensor(2),
        torch.tensor(3.0),
        torch.tensor(10.0),
    ]
    n.lrp(torch.tensor(10.0), 3)
    assert [n.neurons[i].relevance for i in range(len(n.neurons))] == [
        torch.tensor(-6.0),
        torch.tensor(16.0),
        torch.tensor(6.0),
        torch.tensor(10.0),
    ]

    n = Network(7, [[4], [4], [5], [5], [6], [6], []])
    n.weights = {
        (0, 4): torch.tensor(6.0, requires_grad=True),
        (1, 4): torch.tensor(5.0, requires_grad=True),
        (2, 5): torch.tensor(4.0, requires_grad=True),
        (3, 5): torch.tensor(6.0, requires_grad=True),
        (4, 6): torch.tensor(4.0, requires_grad=True),
        (5, 6): torch.tensor(3.0, requires_grad=True),
    }
    n.forward({0: 1, 1: -1, 2: 2, 3: -1, 4: 1, 5: 1, 6: 1})
    print([n.neurons[i].value for i in range(len(n.neurons))])
    n.lrp(torch.tensor(10.0), 6)
    print([n.neurons[i].relevance for i in range(len(n.neurons))])
    # assert False
