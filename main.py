import argparse
import sys

import torch
import torch.nn.functional as F

from crm.core import Network
from crm.utils import get_metrics, make_dataset_cli, train


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="CRM; Example: python3 main.py -f inp.file -o out.file -e 20"
    )
    parser.add_argument("-f", "--file", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    parser.add_argument(
        "-e", "--epochs", type=int, help="number of epochs", required=True
    )
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
    args = cmd_line_args()
    sys.stdout = Logger(args.output)
    file_name = args.file
    with open(file_name, "r") as f:
        graph_file = f.readline()[:-1]
        train_file = f.readline()[:-1]
        test_files = f.readline()[:-1].split()
    X_train, y_train, test_dataset, adj_list = make_dataset_cli(
        graph_file, train_file, test_files
    )
    n = Network(len(adj_list), adj_list)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(n.parameters(), lr=0.01)
    train_losses, train_accs = train(
        n, X_train, y_train, args.epochs, optimizer, criterion, verbose=True
    )
    print("Test Metrics")
    for X_test, y_test in test_dataset:
        print(get_metrics(n, X_test, y_test))
        print("##############################")


if __name__ == "__main__":
    main()
