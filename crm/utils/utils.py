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
        y_pred.append(torch.argmax(n.forward(inp)))
        n.reset()
    return classification_report(
        torch.stack(y_test).numpy(),
        torch.tensor(y_pred).numpy(),
        digits=4,
        output_dict=output_dict,
        zero_division=1,
    )


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
