"""
Tests for the entropy function
"""
# tableStack_test.py
import pandas as pd
import numpy as np
#from pyepidisplay.tableStack import tableStack
from pyepidisplay.data import data
from pyepidisplay.datasets import DATA_PATH
import pytest

# Smoke tests
def test_smoke():
    data("Outbreak")
    return

def test_smoke_no_quote():
    data(Outbreak)
    return

def test_smoke_small():
    data(outbreak)
    return

def test_smoke_cap():
    data(OUTBREAK)
    return

def test_smoke_cap_str():
    data("OUTBREAK")
    return

# one shot test
def test_one_shot_known_column():
    df = data("Outbreak")
    assert df.columns[0] == "id"

## Edge test
def test_wrong_dataset():
    with pytest.raises(
        ValueError, match="Dataset 'Outbreak_abc' not found."
    ):
        data("Outbreak_abc")
    return


# Pattern test
def test_compare_python_r_outbreak_rpy2():
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import numpy as np
    dataset_name = "Outbreak"

    # 1) Load Python dataset
    df_py = data(dataset_name)
    assert isinstance(df_py, pd.DataFrame)

    # 2) Load R and import EpiDisplay
    robjects.r('suppressMessages(library(epiDisplay))')
    robjects.r(f'data({dataset_name})')

    # 3) Convert R data.frame to pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_r = robjects.r[dataset_name]
        df_r = pd.DataFrame(df_r)

    # 4) Compare shape
    assert df_py.shape == df_r.shape, f"Shape mismatch: Python={df_py.shape}, R={df_r.shape}"

    # 5) Compare column names
    assert list(df_py.columns) == list(df_r.columns), \
        f"Column names mismatch: Python={list(df_py.columns)}, R={list(df_r.columns)}"

    # 6) Compare dtypes
    # Map R types to numpy/pandas dtypes
    r_dtypes = df_r.dtypes.apply(lambda x: x.name)
    py_dtypes = df_py.dtypes.apply(lambda x: x.name)
    assert list(py_dtypes) == list(r_dtypes), \
        f"Dtype mismatch: Python={list(py_dtypes)}, R={list(r_dtypes)}"

    # 7) Compare values
    py_values = df_py.to_numpy()
    r_values = df_r.to_numpy()

    # Numeric mask
    numeric_mask = np.vectorize(lambda x: isinstance(x, (int, float, np.number)))
    py_numeric_mask = numeric_mask(py_values)
    r_numeric_mask = numeric_mask(r_values)
    numeric_mask_final = py_numeric_mask & r_numeric_mask

    # Compare numeric values
    if np.any(numeric_mask_final):
        assert np.allclose(py_values[numeric_mask_final], r_values[numeric_mask_final], equal_nan=True), \
            "Numeric values mismatch between Python and R"

    # Compare non-numeric values exactly
    if np.any(~numeric_mask_final):
        assert np.array_equal(py_values[~numeric_mask_final], r_values[~numeric_mask_final]), \
            "Non-numeric values mismatch between Python and R"

# ----------------------------------------------------------
def test_pattern_all_datasets_not_empty():
    dataset_names = data()  # returns list of dataset names
    for name in dataset_names:
        df = data(name)  # load dataset
        assert df.shape[0] > 0, f"{name} has 0 rows"
        assert df.shape[1] > 0, f"{name} has 0 columns"
