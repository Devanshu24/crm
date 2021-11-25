import torch

from crm.core import Network


def get_explanations(n: Network, X_test, y_test, k=3, verbose=False):
    tp = torch.zeros(n.num_neurons)
    fp = torch.zeros(n.num_neurons)
    tn = torch.zeros(n.num_neurons)
    fn = torch.zeros(n.num_neurons)
    for i in range(len(X_test)):
        n.reset()
        pred = torch.argmax(n.forward(X_test[i]))
        if pred == 1:
            n.lrp(torch.tensor(100.0), n.num_neurons - 1)
        else:
            n.lrp(torch.tensor(100.0), n.num_neurons - 2)
        if pred == 1 and y_test[i] == 1:
            tp += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
        if pred == 1 and y_test[i] == 0:
            fp += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
        if pred == 0 and y_test[i] == 0:
            tn += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
        if pred == 0 and y_test[i] == 1:
            fn += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
        rels = []
        for j in range(n.num_neurons):
            if n.neurons[j].successor_neurons == [n.num_neurons - 2, n.num_neurons - 1]:
                rels.append((n.neurons[j].relevance.item(), j))
        rels.sort(reverse=True)
        if verbose:
            print(
                f"{i}: pred = {pred.item()}, true: {y_test[i].item()}, top-{k}: {rels[:k]}"
            )
    tp_values, tp_indices = tp.sort(descending=True)
    fp_values, fp_indices = fp.sort(descending=True)
    tn_values, tn_indices = tn.sort(descending=True)
    fn_values, fn_indices = fn.sort(descending=True)

    tp_rels = []
    for j in range(len(tp_indices)):
        if n.neurons[tp_indices[j]].successor_neurons == [
            n.num_neurons - 2,
            n.num_neurons - 1,
        ]:
            tp_rels.append((tp_values[j].item(), tp_indices[j].item()))
    print(f"TP (top-{k}): {tp_rels[:k]}")

    fp_rels = []
    for j in range(len(fp_indices)):
        if n.neurons[fp_indices[j]].successor_neurons == [
            n.num_neurons - 2,
            n.num_neurons - 1,
        ]:
            fp_rels.append((fp_values[j].item(), fp_indices[j].item()))
    print(f"FP (top-{k}): {fp_rels[:k]}")

    tn_rels = []
    for j in range(len(tn_indices)):
        if n.neurons[tn_indices[j]].successor_neurons == [
            n.num_neurons - 2,
            n.num_neurons - 1,
        ]:
            tn_rels.append((tn_values[j].item(), tn_indices[j].item()))
    print(f"TN (top-{k}): {tn_rels[:k]}")

    fn_rels = []
    for j in range(len(fn_indices)):
        if n.neurons[fn_indices[j]].successor_neurons == [
            n.num_neurons - 2,
            n.num_neurons - 1,
        ]:
            fn_rels.append((fn_values[j].item(), fn_indices[j].item()))
    print(f"FN (top-{k}): {fn_rels[:k]}")
