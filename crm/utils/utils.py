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
