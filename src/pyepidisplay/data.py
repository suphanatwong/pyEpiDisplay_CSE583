import os
import pandas as pd
import builtins
from pyepidisplay.datasets import DATA_PATH


# -------------------------------------------------------------------------
# Register dataset name constants globally so tests like data(Outbreak) work
# -------------------------------------------------------------------------
_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".csv")]

for _filename in _files:
    _base = os.path.splitext(_filename)[0]  # "Outbreak"
    for _variant in {_base, _base.lower(), _base.upper(), _base.capitalize()}:
        setattr(builtins, _variant, _base)   # <-- GLOBAL (no NameError)
# -------------------------------------------------------------------------


def data(name: str = None):
    """Load dataset by name."""
    files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".csv")]

    # no name → return list
    if name is None:
        return [os.path.splitext(f)[0] for f in files]

    name = str(name).lower()
    lookup = {os.path.splitext(f)[0].lower(): f for f in files}

    if name not in lookup:
        raise ValueError(
            f"Dataset '{name}' not found.\n"
            f"Available datasets: {', '.join(os.path.splitext(f)[0] for f in files)}"
        )

    filepath = os.path.join(DATA_PATH, lookup[name])
    return pd.read_csv(filepath)
