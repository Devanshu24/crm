import pytest
import torch
import torch.nn.functional as F

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
        (0, 1): torch.tensor(1.0, requires_grad=True),
        (1, 2): torch.tensor(2.0, requires_grad=True),
        (2, 3): torch.tensor(3.0, requires_grad=True),
    }
    print(n.weights)
    print(n.weights[(2, 3)].grad)
    o = n.forward({0: 1, 1: 1, 2: 1, 3: 1})
    # n.backward({0: 1, 1: 1, 2: 1, 3: 1}, 0.01, 10, {3: lambda x: 0.1 * x})
    print(o)
    o.backward()
    print(n.weights[(2, 3)].grad)
    assert False
    assert n.neurons[3].grad == torch.tensor(0.6)
    assert n.weights[(2, 3)] == torch.tensor(3 - 0.01 * 0.6 * 2)

    n = Network(3, [[1, 2], [2], []])
    n.weights = {
        (0, 1): torch.tensor(1),
        (0, 2): torch.tensor(2),
        (1, 2): torch.tensor(3),
    }
    n.forward({0: 1, 1: 1, 2: 1})
    n.backward({0: 1, 1: 1, 2: 1}, 0.01, 10, {2: lambda x: 0.1 * x})
    assert n.neurons[2].grad == torch.tensor(0.5)
    assert n.neurons[1].grad == torch.tensor(1.5)
    assert n.neurons[0].grad == torch.tensor(2.5)


def test_no_dangling_backward():
    n = Network(4, [[1], [2], [3], []])
    with pytest.raises(Exception):
        n.backward({0: 1, 1: 1, 2: 1, 3: 1}, 0.1, 10, {3: lambda x: 0.1 * x})


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

    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == [torch.tensor(6)]
    n.reset()
    n.set_neuron_activation(
        3, lambda x: torch.exp(x.float()), lambda x: torch.exp(x.float())
    )
    assert n.forward({0: 1, 1: 1, 2: 1, 3: 1}) == [torch.exp(torch.tensor(6).float())]


def test_learning():
    n = Network(
        2,
        [[1], []],
        custom_activations=((lambda x: x, lambda x: 1), (lambda x: x, lambda x: 1)),
    )
    inputs = torch.linspace(-1, 1, 1000)
    labels = inputs / 2
    losses = []
    weights = []
    print(n.weights)
    for i in range(1000):
        out = n.forward({0: inputs[i], 1: 1})
        loss = F.mse_loss(out[0].reshape(1), labels[i].reshape(1))
        losses.append(loss.item())
        cur_weights = n.weights[(0, 1)].item()
        weights.append(cur_weights)
        n.backward({0: inputs[i], 1: 1}, 0.1, loss, {1: lambda x: 2 * (x - labels[i])})
        n.reset()
        assert torch.allclose(
            n.weights[(0, 1)],
            cur_weights - 0.1 * (2 * (out[0] - labels[i]) * inputs[i]),
        )
    # plt.plot(losses)
    # plt.show()
    # plt.plot(weights)
    # plt.show()
    assert torch.allclose(n.forward({0: 7, 1: 1})[0], torch.tensor(3.5))
