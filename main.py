import argparse
import sys

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from crm.core import Network
from crm.utils import (  # get_explanations,; train_distributed,
    get_best_config,
    get_max_explanations,
    get_metrics,
    get_predictions,
    load_object,
    make_dataset_cli,
    seed_all,
    train,
    train_distributed,
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
        "-p", "--predict", help="get predictions for a test set", action="store_true"
    )
    parser.add_argument(
        "-e", "--explain", help="get explanations for predictions", action="store_true"
    )
    parser.add_argument(
        "-t", "--tune", help="tune the hyper parameters", action="store_true"
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
    torch.set_num_threads(16)
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

    # Create CRM structure and train with input data
    print("***Creating CRM structure***")
    n = Network(len(adj_list), adj_list)
    n.to(device)

    if args.saved_model:
        print("***Loading Saved Model***")
        n = load_object(args.saved_model)

    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    if args.tune:
        print("***Get Best Config***")
        best = get_best_config(
            n, X_train, y_train, args.num_epochs, optimizer, criterion
        )
        print(best)

    print("***Training CRM***")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=24, stratify=y_train
    )

    # train_distributed(
    #     n,
    #     X_train,
    #     y_train,
    #     args.num_epochs,
    #     optimizer,
    #     criterion,
    #     X_val,
    #     y_val,
    #     num_workers=16,
    # )

    train_losses, train_accs, val_losses, val_accs = train(
        n,
        X_train,
        y_train,
        args.num_epochs,
        torch.optim.Adam(n.parameters(), lr=best["lr"] if args.tune else 0.001),
        criterion,
        X_val=X_val,
        y_val=y_val,
        save_here=args.output + "_model",
        verbose=args.verbose,
    )

    # Train metrics
    if not args.saved_model:
        print("***Train Metrics***")
        print(get_metrics(n, X_train, y_train))
        print("-------------------------------------")

    # Test metrics
    print("***Test Metrics***")
    for X_test, y_test in test_dataset:
        print(get_metrics(n, X_test, y_test))
        print("-------------------------------------")

    # Predict for the test instances
    if args.predict:
        print("***Predicting the class labels for the test set***")
        for X_test, y_test in test_dataset:
            get_predictions(n, X_test, y_test)

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
                k=1,
                verbose=args.verbose,
            )
            print("-------------------------------------")


if __name__ == "__main__":
    """
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
    """
    main()
