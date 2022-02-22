import argparse
import sys

import torch
import torch.nn.functional as F

from crm.core import Network
from crm.utils import get_explanations, get_metrics, make_dataset_cli, seed_all, train


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="CRM; Example: python3 main.py -f inp.file -o out.file -n 20"
    )
    parser.add_argument("-f", "--file", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
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
    file_name = args.file
    with open(file_name, "r") as f:
        graph_file = f.readline()[:-1]
        train_file = f.readline()[:-1]
        test_files = f.readline()[:-1].split()
        true_explanations = list(map(int, f.readline()[:-1].split()))
    X_train, y_train, test_dataset, adj_list, edges = make_dataset_cli(
        graph_file, train_file, test_files, device=device
    )
    n = Network(len(adj_list), adj_list)
    n.to(device)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    train_losses, train_accs = train(
        n, X_train, y_train, args.num_epochs, optimizer, criterion, verbose=args.verbose
    )

    # Train metrics
    print("Train Metrics")
    print(get_metrics(n, X_train, y_train))
    print("##############################")

    # Test metrics
    print("Test Metrics")
    for X_test, y_test in test_dataset:
        print(get_metrics(n, X_test, y_test))
        print("-------------------------------------")
    print("##############################")

    if args.explain:
        print("Explanations")
        for X_test, y_test in test_dataset:
            get_explanations(
                n,
                X_test,
                y_test,
                true_explanations=true_explanations,
                verbose=args.verbose,
            )
            print("-------------------------------------")
        print("##############################")


if __name__ == "__main__":
    main()
