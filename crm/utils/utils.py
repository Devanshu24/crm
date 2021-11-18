import dill


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as inp:
        return dill.load(inp)
