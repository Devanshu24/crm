import argparse
import sys

import torch
import torch.nn.functional as F

from crm.core import Network
from crm.utils import (  # get_explanations,
    get_max_explanations,
    get_metrics,
    load_object,
    make_dataset_cli,
    seed_all,
    train,
)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="CRM; Example: python3 main.py -f inp.file -o out.file -n 20"
    )
    parser.add_argument("-f", "--file", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    parser.add_argument(
        "-s",
        "--saved-model",
        type=str,
        help="location of saved model",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-n", "--num-epochs", type=int, help="number of epochs", required=True
    )
    parser.add_argument(
        "-e", "--explain", help="get explanations for predictions", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="get verbose outputs", action="store_true"
    )
    parser.add_argument("-g", "--gpu", help="run model on gpu", action="store_true")
    args = parser.parse_args()
    return args


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    seed_all(24)
    args = cmd_line_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    sys.stdout = Logger(args.output)
    print(args)

    # Load data
    file_name = args.file
    print("***Loading data***")
    with open(file_name, "r") as f:
        graph_file = f.readline()[:-1]
        train_file = f.readline()[:-1]
        test_files = f.readline()[:-1].split()
        true_explanations = list(map(int, f.readline()[:-1].split()))
    X_train, y_train, test_dataset, adj_list, edges = make_dataset_cli(
        graph_file, train_file, test_files, device=device
    )
    # layer_one_neurons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # num_neurons = max(layer_one_neurons)+1
    # e1 = []
    # for e in edges:
    #     if e[0] in layer_one_neurons and e[1] in layer_one_neurons:
    #         e1.append(e)
    # adj_list = [[] for i in range(num_neurons)]
    # for u, v in e1:
    #     adj_list[u].append(v)
    # n = Network(num_neurons, adj_list)
    # orig_output_neurons = n.output_neurons
    # adj_list.append([])
    # adj_list.append([])
    # num_neurons = len(adj_list)
    # for i in range(num_neurons):
    #     if i in orig_output_neurons:
    #         adj_list[i].append(num_neurons - 2)
    #         adj_list[i].append(num_neurons - 1)
    # for i in range(len(X_train)):
    #     X_train[i][num_neurons - 2] = 1
    #     X_train[i][num_neurons - 1] = 1
    # for i in range(len(test_dataset)):
    #     for j in range(len(test_dataset[i][0])):
    #         test_dataset[i][0][j][num_neurons - 2] = 1
    #         test_dataset[i][0][j][num_neurons - 1] = 1

    # Create CRM structure and train with input data
    print("***Creating CRM structure***")
    n = Network(len(adj_list), adj_list)
    n.to(device)

    if args.saved_model:
        print("***Loading Saved Model***")
        n = load_object(args.saved_model)

    print("***Training CRM***")
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    train_losses, train_accs = train(
        n,
        X_train,
        y_train,
        args.num_epochs,
        optimizer,
        criterion,
        args.output + "_model",
        verbose=args.verbose,
    )

    # Train metrics
    print("***Train Metrics***")
    print(get_metrics(n, X_train, y_train))
    print("-------------------------------------")

    # Test metrics
    print("***Test Metrics***")
    for X_test, y_test in test_dataset:
        print(get_metrics(n, X_test, y_test))
        print("-------------------------------------")

    # Explain the test instances
    if args.explain:
        print("***Generating explanations for the test set***")
        for X_test, y_test in test_dataset:
            # get_explanations(
            #    n,
            #    X_test,
            #    y_test,
            #    true_explanations=true_explanations,
            #    k=1,
            #    verbose=args.verbose
            # )

            # added by T: get max explanations
            get_max_explanations(
                n,
                X_test,
                y_test,
                true_explanations=true_explanations,
                verbose=args.verbose,
            )
            print("-------------------------------------")


if __name__ == "__main__":
    main()
