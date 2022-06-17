import random

import numpy as np
import ray
import torch
import torch.distributed.autograd as dist_autograd
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange

from crm.core import Network
from crm.distributed import DataWorker, ParameterServer
from crm.utils import get_metrics, save_object


def train_distributed(
    n: Network,
    X_train,
    y_train,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion,
    X_val,
    y_val,
    num_workers: int,
    verbose: bool = True,
):
    raise NotImplementedError("ToDo")
    iterations = 10
    test_loader = zip(X_val, y_val)  # noqa
    print("Running Asynchronous Parameter Server Training.")

    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-3, n.num_neurons, n.adj_list)
    workers = [
        DataWorker.remote(X_train, y_train, n.num_neurons, n.adj_list)
        for i in range(num_workers)
    ]

    current_weights = ps.get_weights.remote()

    gradients = {}
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    for i in range(iterations * num_workers):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            pass
        # Evaluate the current model after every 10 updates.
        n.set_weights(ray.get(current_weights))
        accuracy = get_metrics(n, X_val, y_val, output_dict=True)["accuracy"]
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))


def train(
    n: Network,
    X_train,
    y_train,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion,
    save_here: str = None,
    X_val=None,
    y_val=None,
    verbose: bool = False,
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    min_loss = 1e10
    for e in trange(num_epochs):
        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)
        local_train_losses = []
        for i in trange(len(X_train)):
            # print(f"Epoch {e}/{num_epochs} | Batch {i}/{len(X_train)}")
            f_mapper = X_train[i]
            out = n.forward(f_mapper).reshape(1, -1)
            loss = criterion(out, y_train[i].reshape(1))
            local_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n.reset()
        with torch.no_grad():
            train_losses.append(sum(local_train_losses) / len(local_train_losses))
            train_accs.append(
                get_metrics(n, X_train, y_train, output_dict=True)["accuracy"]
            )
            if X_val is not None and y_val is not None:
                local_val_losses = []
                for j in range(len(X_val)):
                    f_mapper = X_val[j]
                    out = n.forward(f_mapper).reshape(1, -1)
                    loss = criterion(out, y_val[j].reshape(1))
                    local_val_losses.append(loss.item())
                    n.reset()
                val_losses.append(sum(local_val_losses) / len(local_val_losses))
                val_accs.append(
                    get_metrics(n, X_val, y_val, output_dict=True)["accuracy"]
                )
                if val_losses[-1] < min_loss:
                    min_loss = val_losses[-1]
                    patience = 0
                else:
                    patience += 1
                if patience > 3:
                    print("Patience exceeded. Stopping training.")
                    break
        if verbose:
            tqdm.write(f"Epoch {e}")
            tqdm.write(f"Train loss: {train_losses[-1]}")
            tqdm.write(f"Train acc: {train_accs[-1]}")
            if X_val is not None and y_val is not None:
                tqdm.write(f"Val loss: {val_losses[-1]}")
                tqdm.write(f"Val acc: {val_accs[-1]}")
            tqdm.write("-------------------------------------")
        if save_here is not None:
            save_object(n, f"{save_here}_{e}.pt")
    return (
        (train_losses, train_accs, val_losses, val_accs)
        if X_val is not None and y_val is not None
        else (train_losses, train_accs)
    )


def get_best_config(
    n: Network,
    X,
    y,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion,
):
    """Uses Ray Tune and Optuna to find the best configuration for the network."""

    def train_with_config(config):
        """Train the network with the given config."""
        optimizer = torch.optim.Adam(n.parameters(), lr=config["lr"])
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=24, stratify=y
        )
        train_losses, train_accs, val_losses, val_accs = train(
            n=n,
            X_train=X_train,
            y_train=y_train,
            num_epochs=num_epochs,
            optimizer=optimizer,
            criterion=criterion,
            X_val=X_val,
            y_val=y_val,
            verbose=False,
        )
        return {
            "mean_train_loss": np.mean(train_losses),
            "mean_train_acc": np.mean(train_accs),
            "mean_val_loss": np.mean(val_losses),
            "mean_val_acc": np.mean(val_accs),
        }

    config = {"lr": tune.grid_search([0.01, 0.001, 0.005, 0.0001])}
    algo = BasicVariantGenerator(max_concurrent=16)
    # uncomment and set max_concurrent to limit number of cores
    # algo = ConcurrencyLimiter(algo, max_concurrent=16)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(
        train_with_config,
        num_samples=1,
        config=config,
        name="optuna_train",
        metric="mean_val_acc",
        mode="max",
        search_alg=algo,
        scheduler=scheduler,
        verbose=0,
        max_failures=1,
    )

    return analysis.best_config
