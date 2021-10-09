import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from crm import Network

if __name__ == "__main__":
    n = Network(
        2,
        [[1], []],
        custom_activations=((lambda x: x, lambda x: 1), (lambda x: x, lambda x: 1)),
    )
    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    inputs = torch.linspace(-1, 1, 1000)
    labels = inputs / 2
    losses = []
    for i in range(1000):
        out = n.forward({0: inputs[i], 1: 1})
        loss = F.mse_loss(out[0].reshape(1), labels[i].reshape(1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        n.reset()
    print(n.weights)
    plt.plot(losses)
    plt.show()
