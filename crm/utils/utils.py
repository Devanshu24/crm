import random

import dill
import torch
from sklearn.metrics import classification_report


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as inp:
        return dill.load(inp)


def get_metrics(n, X_test, y_test, output_dict=False):
    y_pred = []
    for inp in X_test:
        # print(inp)
        y_pred.append(torch.argmax(n.forward(inp)))
        n.reset()
    return classification_report(
        torch.stack(y_test).numpy(),
        torch.tensor(y_pred).numpy(),
        digits=4,
        output_dict=output_dict,
        zero_division=1,
    )


def get_predictions(n, X_test, y_test):
    print(f"Making predictions for {len(X_test)} test instances::")
    print("Instance: y,y_pred,y_pred_prob,oth_pred,oth_pred_prob")

    i = 0
    for (x, y) in list(zip(X_test, y_test)):
        n.reset()
        preds = n.forward(x)
        preds.cpu().detach()
        if not torch.sum(preds):
            preds = torch.tensor([1, 0])
        preds = preds / torch.sum(preds)
        y_pred_prob = torch.max(preds)
        y_pred = torch.argmax(preds)
        # compute prob for the other class
        if y_pred:
            oth_pred = 0
        else:
            oth_pred = 1
        oth_pred_prob = 1 - y_pred_prob
        print(
            f"Inst {i}: {y},{y_pred},{y_pred_prob:.6f},{oth_pred},{oth_pred_prob:.6f}"
        )
        i = i + 1


def seed_all(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)
    except ImportError:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
