import torch
from tqdm.auto import tqdm, trange

from crm.core import Network
from crm.utils import get_metrics


def train(
    n: Network,
    X_train,
    y_train,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion,
    X_val=None,
    y_val=None,
    verbose: bool = False,
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for e in trange(num_epochs):
        idx = torch.randperm(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]
        local_train_losses = []
        for i in trange(len(X_train)):
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
                    out = n.forward(X_val).reshape(1, -1)
                    loss = criterion(out, y_val[j].reshape(1))
                    local_val_losses.append(loss.item())
                    n.reset()
                val_losses.append(sum(local_val_losses) / len(local_val_losses))
                val_accs.append(
                    get_metrics(n, X_val, y_val, output_dict=True)["accuracy"]
                )
        if verbose:
            tqdm.write(f"Epoch {e}")
            tqdm.write(f"Train loss: {train_losses[-1]}")
            tqdm.write(f"Train acc: {train_accs[-1]}")
            if X_val is not None and y_val is not None:
                tqdm.write(f"Val loss: {val_losses[-1]}")
                tqdm.write(f"Val acc: {val_accs[-1]}")
            tqdm.write("##############################")
    return (
        (train_losses, train_accs, val_losses, val_accs)
        if X_val is not None and y_val is not None
        else (train_losses, train_accs)
    )
