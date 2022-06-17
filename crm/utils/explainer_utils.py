import torch
from tqdm.auto import tqdm

from crm.core import Network


def get_explanations(
    n: Network, X_test, y_test, true_explanations, k=3, verbose=False, all_layers=True
):
    tp = torch.zeros(n.num_neurons)
    fp = torch.zeros(n.num_neurons)
    tn = torch.zeros(n.num_neurons)
    fn = torch.zeros(n.num_neurons)

    tp_scores, tp_count = 0, 0
    fp_scores, fp_count = 0, 0
    tn_scores, tn_count = 0, 0
    fn_scores, fn_count = 0, 0

    for i in tqdm(range(len(X_test)), desc="Explanations XTest"):
        n.reset()
        pred = torch.argmax(n.forward(X_test[i]))
        if pred == 1:
            n.lrp(torch.tensor(100.0), n.num_neurons - 1)
        else:
            n.lrp(torch.tensor(100.0), n.num_neurons - 2)
        rels = [[] * n.num_layers for _ in range(n.num_layers)]
        for j in range(n.num_neurons):
            if n.neurons[j] not in [n.num_neurons - 2, n.num_neurons - 1]:
                try:
                    rels[n.neurons[j].layer].append(
                        (torch.tensor(n.neurons[j].relevance).item(), j)
                    )
                except Exception as e:
                    print(j, n.neurons[j].layer, n.neurons[j].relevance)
                    print(e)
                    assert False

        rels = [sorted(x, key=lambda x: x, reverse=True) for x in rels]
        if verbose:
            if all_layers:
                print(f"{i}: pred: {pred}, true: {y_test[i]}")
                for l_id in range(n.num_layers):
                    print(f"top-{k}-L{l_id}: {rels[l_id][:k]}")
            else:
                print(
                    f"{i}: pred = {pred.item()}, true: {y_test[i].item()}, top-{k}: {rels[n.num_layers-1][:k]}"
                )
            print("-" * 20)
        if pred == 1 and y_test[i] == 1:
            tp += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
            tp_scores += (
                1
                if len(
                    list(
                        set(true_explanations)
                        & set(
                            [
                                rels[n.num_layers - 1][j][1]
                                for j in range(min(k, len(rels[n.num_layers - 1])))
                            ]
                        )
                    )
                )
                > 0
                else 0
            )
            tp_count += 1
        if pred == 1 and y_test[i] == 0:
            fp += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
            fp_scores += (
                1
                if len(
                    list(
                        set(true_explanations)
                        & set(
                            [
                                rels[n.num_layers - 1][j][1]
                                for j in range(min(k, len(rels[n.num_layers - 1])))
                            ]
                        )
                    )
                )
                > 0
                else 0
            )
            fp_count += 1
        if pred == 0 and y_test[i] == 0:
            tn += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
            tn_scores += (
                1
                if len(
                    list(
                        set(true_explanations)
                        & set(
                            [
                                rels[n.num_layers - 1][j][1]
                                for j in range(min(k, len(rels[n.num_layers - 1])))
                            ]
                        )
                    )
                )
                > 0
                else 0
            )
            tn_count += 1
        if pred == 0 and y_test[i] == 1:
            fn += torch.tensor([n.neurons[i].relevance for i in range(n.num_neurons)])
            fn_scores += (
                1
                if len(
                    list(
                        set(true_explanations)
                        & set(
                            [
                                rels[n.num_layers - 1][j][1]
                                for j in range(min(k, len(rels[n.num_layers - 1])))
                            ]
                        )
                    )
                )
                > 0
                else 0
            )
            fn_count += 1

    print("SCORES")
    print(f"TP:{tp_scores}/{tp_count}")
    print(f"FP:{fp_scores}/{fp_count}")
    print(f"TN:{tn_scores}/{tn_count}")
    print(f"FN:{fn_scores}/{fn_count}")
    print("####################################")

    print("SUMMED RELS")

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


# added by T:BFS to get the ancestors of neurons
def get_ancestors_of_neurons(n: Network, current_neurons):
    visited = list(current_neurons)
    queue = list(current_neurons)
    while queue:
        visit = queue.pop(0)
        for predecessor_neuron in n.neurons[visit].predecessor_neurons:
            if predecessor_neuron not in visited:
                visited.append(predecessor_neuron)
                queue.append(predecessor_neuron)
    return visited


# added by T: Get maximal explanation of a CRM
def get_max_explanations(
    n: Network, X_test, y_test, true_explanations, k=1, verbose=False
):
    tp_count = 0
    fn_count = 0
    tn_count = 0
    fp_count = 0

    cep_count = 0  # correctly explained positives
    cen_count = 0  # correctly explained negatives
    iep_count = 0  # incorrectly explained positives
    ien_count = 0  # incorrectly explained negatives

    print(f"Explaining {len(X_test)} test instances::")
    print(
        f"Instance: y,y_pred,tp_count,fn_count,tn_count,fp_count,Top-{k}_neurons,cep_count,cen_count,iep_count,ien_count"
    )
    for i in tqdm(range(len(X_test)), desc="Explaining X_test"):
        n.reset()
        pred = torch.argmax(n.forward(X_test[i]))
        if pred == 1:
            n.lrp(torch.tensor(100.0), n.num_neurons - 1)
        else:
            n.lrp(torch.tensor(100.0), n.num_neurons - 2)

        rels = [[] * n.num_layers for _ in range(n.num_layers)]
        for j in range(n.num_neurons):
            if n.neurons[j] not in [n.num_neurons - 2, n.num_neurons - 1]:
                try:
                    rels[n.neurons[j].layer].append(
                        (round(torch.tensor(n.neurons[j].relevance).item(), 4), j)
                    )
                except Exception as e:
                    print(j, n.neurons[j].layer, n.neurons[j].relevance)
                    print(e)
                    assert False

        rels = [sorted(x, key=lambda x: x, reverse=True) for x in rels]
        topk_rels = rels[n.num_layers - 2][:k]
        topk_vertices = [x[1] for x in topk_rels]
        ancestors = get_ancestors_of_neurons(n, topk_vertices)

        # obtain eval metrics: accuracy and fidelity
        if y_test[i] == 1:
            if pred == 1:
                tp_count += 1
            else:
                fn_count += 1

            if set(true_explanations) & set(ancestors):
                cep_count += 1
            else:
                iep_count += 1

        if y_test[i] == 0:
            if pred == 0:
                tn_count += 1
            else:
                fp_count += 1

            if set(true_explanations) & set(ancestors):
                ien_count += 1
            else:
                cen_count += 1

        print(  # noqa
            f"Inst {i}: {y_test[i]},{pred},{tp_count},{fn_count},{tn_count},{fp_count},{topk_vertices},{cep_count},{cen_count},{iep_count},{ien_count}"  # noqa
        )  # noqa
        print(f"\tAncestors of {topk_vertices}: {ancestors}")
        print("\tTop-5 neurons in each CRM layer (ordered by relevance, descending):")
        for l_id in reversed(range(n.num_layers)):
            print(f"\t\tL{l_id}: {rels[l_id][:5]}")

    accuracy = (tp_count + tn_count) / (tp_count + fn_count + tn_count + fp_count)
    fidelity = (cep_count + cen_count) / (cep_count + cen_count + iep_count + ien_count)

    print("\n-------------------------------------")
    print("Explanation statistics:")
    print("-------------------------------------")
    print(f"TP: {tp_count}")
    print(f"FN: {fn_count}")
    print(f"TN: {tn_count}")
    print(f"FP: {fp_count}")
    print(f"CEP: {cep_count}")
    print(f"CEN: {cen_count}")
    print(f"IEP: {iep_count}")
    print(f"IEN: {ien_count}")
    print(f"Accuracy: {accuracy}, Fidelity: {fidelity}")
