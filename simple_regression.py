import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from crm.core import Network


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n = Network(
        2,
        [[1], []],
        custom_activations=((lambda x: x, lambda x: 1), (lambda x: x, lambda x: 1)),
    )
    n.to(device)
    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    inputs = torch.linspace(-1, 1, 1000).to(device)
    labels = inputs / 2
    print(labels[0])
    losses = []
    for i in range(1000):
        out = n.forward([inputs[i], 1])
        loss = F.mse_loss(out[0].reshape(1), labels[i].reshape(1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        n.reset()
    print(n.weights)
    plt.plot(losses)
    plt.show()
