import random

import torch
from tqdm.auto import trange

from crm.core import Network


def train(
    n: Network,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion,
    verbose: bool = False,
):
    losses = []
    for e in trange(num_epochs):
        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)
        for i in trange(len(X_train)):
            f_mapper = X_train[i]
            out = n.forward(f_mapper).reshape(1, -1)
            loss = criterion(out, y_train[i].reshape(1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n.reset()
    return losses
