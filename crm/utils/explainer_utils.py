import torch

from crm.core import Network


def get_explanations(n: Network, X_test, y_test, k=3):
    for i in range(len(X_test)):
        n.reset()
        pred = torch.argmax(n.forward(X_test[i]))
        if pred:
            n.lrp(torch.tensor(100.0), n.num_neurons - 1)
        else:
            n.lrp(torch.tensor(100.0), n.num_neurons - 2)
        rels = []
        for j in range(n.num_neurons):
            if n.neurons[j].successor_neurons == [n.num_neurons - 2, n.num_neurons - 1]:
                rels.append((n.neurons[j].relevance.item(), j))
        rels.sort(reverse=True)
        print(
            f"{i}: pred = {pred.item()}, true: {y_test[i].item()}, top-{k}: {rels[:k]}"
        )
