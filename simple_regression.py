import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from crm import Network

if __name__ == "__main__":
    n = Network(2, [[1], []])
    inputs = torch.linspace(-1, 1, 1000)
    labels = inputs / 2
    losses = []
    weights = []
    for i in range(1000):
        out = n.forward({0: inputs[i], 1: 1})
        loss = F.mse_loss(out[0].reshape(1), labels[i].reshape(1))
        losses.append(loss.item())
        cur_weights = n.weights[(0, 1)].item()
        weights.append(cur_weights)
        n.backward({0: inputs[i], 1: 1}, 0.1, loss, lambda x: 2 * (x - labels[i]))
        n.reset()
        assert torch.allclose(
            n.weights[(0, 1)],
            cur_weights - 0.1 * (2 * (out[0] - labels[i]) * inputs[i]),
        )
    print(n.weights)
    plt.plot(losses)
    plt.show()
