import re

import torch

from crm.core import Network
from crm.utils import save_object


def make_dataset(folder, network_name, save: bool = True):
    """Creates dataset from raw files"""
    graph_file = f"{folder}/raw/{network_name}.pl"
    train_pos_file = f"{folder}/raw/{network_name}_train_features_pos"
    train_neg_file = f"{folder}/raw/{network_name}_train_features_neg"
    test_pos_file = f"{folder}/raw/{network_name}_test_features_pos"
    test_neg_file = f"{folder}/raw/{network_name}_test_features_neg"

    with open(graph_file, "r") as f:
        edges_raw = []
        while True:
            s = f.readline()
            if not s:
                break
            if "connected" in s:
                edges_raw.append(s)

    edges = []
    for i in range(len(edges_raw)):

        # Binary Operators
        b_match = re.search("a\((.*)\),\[a\((.*?)\),a\((.*?)\)\]", edges_raw[i])  # noqa
        u_match = re.search("a\((.*)\),\[a\((.*?)\)\]", edges_raw[i])  # noqa

        end, start_one, start_two = -1, -1, -1

        if b_match:
            end, start_one, start_two = (
                int(b_match.group(1)) - 1,
                int(b_match.group(2)) - 1,
                int(b_match.group(3)) - 1,
            )
        else:
            end, start_one = int(u_match.group(1)) - 1, int(u_match.group(2)) - 1
        # start_one --> end
        # start_two --> end
        if start_one == start_two and start_one != -1:
            edges.append((int(start_one), int(end)))
        else:
            if start_one != -1:
                edges.append((int(start_one), int(end)))
            if start_two != -1:
                edges.append((int(start_two), int(end)))

    with open(f"{folder}/edges.txt", "w") as fp:
        for u, v in edges:
            fp.write(f"{u} {v}\n")

    num_neurons = max([max(u, v) for u, v in edges]) + 1

    X_train = []
    y_train = []

    with open(f"{train_pos_file}", "r") as f:
        while True:
            gg = f.readline().split(" ")[3:-1]
            if not gg:
                break
            all_pos = [int(e) - 1 for e in gg]
            dd = torch.tensor([1. if i in all_pos else 0. for i in range(num_neurons)])
            X_train.append(dd)
            y_train.append(torch.tensor(1))

    with open(f"{train_neg_file}", "r") as f:
        while True:
            gg = f.readline().split(" ")[3:-1]
            if not gg:
                break
            all_pos = [int(e) - 1 for e in gg]
            dd = torch.tensor([1. if i in all_pos else 0. for i in range(num_neurons)])
            X_train.append(dd)
            y_train.append(torch.tensor(0))

    X_train = torch.stack([elem for elem in X_train], dim=0)
    y_train = torch.stack([elem for elem in y_train], dim=0)

    X_test = []
    y_test = []

    with open(f"{test_pos_file}", "r") as f:
        while True:
            gg = f.readline().split(" ")[3:-1]
            if not gg:
                break
            all_pos = [int(e) - 1 for e in gg]
            dd = torch.tensor([1. if i in all_pos else 0. for i in range(num_neurons)])
            X_test.append(dd)
            y_test.append(torch.tensor(1))

    with open(f"{test_neg_file}", "r") as f:
        while True:
            gg = f.readline().split(" ")[3:-1]
            if not gg:
                break
            all_pos = [int(e) - 1 for e in gg]
            dd = torch.tensor([1. if i in all_pos else 0. for i in range(num_neurons)])
            X_test.append(dd)
            y_test.append(torch.tensor(0))

    X_test = torch.stack([elem for elem in X_test], dim=0)
    y_test = torch.stack([elem for elem in y_test], dim=0)

    adj_list = [[] for i in range(num_neurons)]
    for u, v in edges:
        adj_list[u].append(v)

    n = Network(num_neurons, adj_list)
    orig_output_neurons = n.output_neurons
    adj_list.append([])
    adj_list.append([])
    num_neurons = len(adj_list)
    for i in range(num_neurons):
        if i in orig_output_neurons:
            adj_list[i].append(num_neurons - 2)
            adj_list[i].append(num_neurons - 1)

    X_train = torch.hstack([X_train, torch.ones(X_train.shape[0], 2)])
    X_test = torch.hstack([X_test, torch.ones(X_test.shape[0], 2)])
    
    n = Network(num_neurons, adj_list)
    n.forward(X_train[0])
    n.reset()
    n.forward(X_test[0])

    if save:
        save_object(adj_list, f"{folder}/adj_list.dill")
        save_object(X_train, f"{folder}/X_train.dill")
        save_object(X_test, f"{folder}/X_test.dill")
        save_object(y_train, f"{folder}/y_train.dill")
        save_object(y_test, f"{folder}/y_test.dill")

    return X_train, y_train, X_test, y_test, adj_list
