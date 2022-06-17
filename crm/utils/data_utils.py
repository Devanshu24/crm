import os
import re
from typing import List

import torch

from crm.core import Network
from crm.utils import save_object


def make_dataset_cli(
    graph_file: str, train_file: str, test_files: List[str], device=torch.device("cpu")
):
    """
    Create a dataset from a CLI.
    """
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

    num_neurons = max([max(u, v) for u, v in edges]) + 1

    X_train = []
    y_train = []

    train_pos_file = train_file + "_pos"
    train_neg_file = train_file + "_neg"

    if os.path.exists(f"{train_pos_file}"):
        with open(f"{train_pos_file}", "r") as f:
            while True:
                gg = f.readline()  # .split(" ")[3:-1]
                if not gg:
                    break
                gg = gg.split(" ")[3:]
                if not gg:
                    continue
                all_pos = [int(e) - 1 for e in gg if e != "\n"]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_train.append(dd)
                y_train.append(torch.tensor(1))

    if os.path.exists(f"{train_neg_file}"):
        with open(f"{train_neg_file}", "r") as f:
            while True:
                gg = f.readline()  # .split(" ")[3:-1]
                if not gg:
                    break
                gg = gg.split(" ")[3:]
                if not gg:
                    continue
                all_pos = [int(e) - 1 for e in gg if e != "\n"]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_train.append(dd)
                y_train.append(torch.tensor(0))

    test_dataset = []
    for test_file in test_files:
        X_test = []
        y_test = []
        test_pos_file = test_file + "_pos"
        test_neg_file = test_file + "_neg"

        inst_id = 0
        if os.path.exists(f"{test_pos_file}"):
            with open(f"{test_pos_file}", "r") as f:
                while True:
                    gg = f.readline()  # .split(" ")[3:-1]
                    if not gg:
                        break
                    inst_id = inst_id + 1
                    gg = gg.split(" ")[3:]
                    if not gg:
                        print(f"Empty instance ID {inst_id}")
                        continue
                    all_pos = [int(e) - 1 for e in gg if e != "\n"]
                    dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                    X_test.append(dd)
                    y_test.append(torch.tensor(1))
        print(
            f"Loaded {test_pos_file} file with {inst_id} instances (incl. empty instances)."
        )
        num_test_pos = inst_id

        if os.path.exists(f"{test_neg_file}"):
            with open(f"{test_neg_file}", "r") as f:
                while True:
                    gg = f.readline()  # .split(" ")[3:-1]
                    if not gg:
                        break
                    inst_id = inst_id + 1
                    gg = gg.split(" ")[3:]
                    if not gg:
                        print(f"Empty instance ID {inst_id}")
                        continue
                    all_pos = [int(e) - 1 for e in gg if e != "\n"]
                    dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                    X_test.append(dd)
                    y_test.append(torch.tensor(0))
        print(
            f"Loaded {test_neg_file} file with {inst_id - num_test_pos} instances (incl. empty instances)."
        )
        quit()
        test_dataset.append((X_test, y_test))

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

    # # TODO: Verifyyyyy
    # print("Connecting all the input neurons to output also!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for i in range(n.num_neurons):
    #     if len(n.neurons[i].predecessor_neurons) == 0 and n.neurons[
    #         i
    #     ].successor_neurons != [num_neurons - 2, num_neurons - 1]:
    #         adj_list[i].append(num_neurons - 2)
    #         adj_list[i].append(num_neurons - 1)

    for i in range(len(X_train)):
        X_train[i][num_neurons - 2] = 1
        X_train[i][num_neurons - 1] = 1

    for X_test, y_test in test_dataset:
        for i in range(len(X_test)):
            X_test[i][num_neurons - 2] = 1
            X_test[i][num_neurons - 1] = 1

    # for i in range(len(X_train)):
    #     X_train[i] = list(X_train[i].values())
    # X_train = torch.tensor(X_train).to(device)
    # y_train = torch.tensor(y_train).to(device)

    # for i in range(len(test_dataset)):
    #     for j in range(len(test_dataset[i][0])):
    #         test_dataset[i][0][j] = list(test_dataset[i][0][j].values())
    #     test_dataset[i] = (
    #         torch.tensor(test_dataset[i][0]).to(device),
    #         torch.tensor(test_dataset[i][1]).to(device),
    #     )
    return X_train, y_train, test_dataset, adj_list, edges


def make_dataset(folder, network_name, device=torch.device("cpu"), save: bool = False):
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

    if os.path.exists(f"{train_pos_file}"):
        with open(f"{train_pos_file}", "r") as f:
            while True:
                gg = f.readline().split(" ")[3:-1]
                if not gg:
                    break
                all_pos = [int(e) - 1 for e in gg]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_train.append(dd)
                y_train.append(torch.tensor(1))

    if os.path.exists(f"{train_neg_file}"):
        with open(f"{train_neg_file}", "r") as f:
            while True:
                gg = f.readline().split(" ")[3:-1]
                if not gg:
                    break
                all_pos = [int(e) - 1 for e in gg]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_train.append(dd)
                y_train.append(torch.tensor(0))

    X_test = []
    y_test = []

    if os.path.exists(f"{test_pos_file}"):
        with open(f"{test_pos_file}", "r") as f:
            while True:
                gg = f.readline().split(" ")[3:-1]
                if not gg:
                    break
                all_pos = [int(e) - 1 for e in gg]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_test.append(dd)
                y_test.append(torch.tensor(1))

    if os.path.exists(f"{test_neg_file}"):
        with open(f"{test_neg_file}", "r") as f:
            while True:
                gg = f.readline().split(" ")[3:-1]
                if not gg:
                    break
                all_pos = [int(e) - 1 for e in gg]
                dd = {i: 1 if i in all_pos else 0 for i in range(num_neurons)}
                X_test.append(dd)
                y_test.append(torch.tensor(0))

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

    for i in range(len(X_train)):
        X_train[i][num_neurons - 2] = 1
        X_train[i][num_neurons - 1] = 1

    for i in range(len(X_test)):
        X_test[i][num_neurons - 2] = 1
        X_test[i][num_neurons - 1] = 1

    n = Network(num_neurons, adj_list)
    n.forward(X_train[0])
    n.reset()
    n.forward(X_test[0])

    for i in range(len(X_train)):
        X_train[i] = list(X_train[i].values())
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)

    for i in range(len(X_test)):
        X_test[i] = list(X_test[i].values())
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)

    if save:
        save_object(adj_list, f"{folder}/adj_list.dill")
        save_object(X_train, f"{folder}/X_train.dill")
        save_object(X_test, f"{folder}/X_test.dill")
        save_object(y_train, f"{folder}/y_train.dill")
        save_object(y_test, f"{folder}/y_test.dill")

    return X_train, y_train, X_test, y_test, adj_list


def edges_to_adj_list(edges):
    num_neurons = max([max(u, v) for u, v in edges]) + 1
    adj_list = [[] for i in range(num_neurons)]
    for u, v in edges:
        adj_list[u].append(v)
    return adj_list
