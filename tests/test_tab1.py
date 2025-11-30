import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from pyepidisplay.data import data
from pyepidisplay.tab1 import tab1

#outbreak = pd.read_csv("/Users/Joey/Downloads/FallQuarter/CSE583/pyEpiDisplay/src/pyepidisplay/datasets/Outbreak.csv")
outbreak = data("Outbreak")

def test_smoke():
    """
    author: Jiayi
    reviewer: Marthin
    category: smoke test
    """
    tab1("age", outbreak)
    return

def check_output_type():
    """
    author: Jiayi
    reviewer: Marthin
    category: one-shot test
    """
    type(tab1("age", outbreak)) 
    return

def check_column_number():
    """
    author: Jiayi
    reviewer: Marthin
    category: one-shot test
    """
    tab1("age", outbreak).shape[1] 
    return

def valid_input_1():
    """
    author: Jiayi
    reviewer: Marthin
    category: edge test
    """
    with pytest.raises(ValueError, match ="Column name must be a string."):
        tab1(3, outbreak)
    return

def valid_input_2():
    """
    author: Jiayi
    reviewer: Marthin
    category: edge test
    """
    with pytest.raises(ValueError, match ="Input data must be a pandas DataFrame." ):
        tab1("age", outbreak)
    return

def column_exists():
    """
    author: Jiayi
    reviewer: Marthin
    category: edge test
    """
    with pytest.raises(ValueError, match = "Column is not found in DataFrame."):
        tab1("region", outbreak)
    return

def check_null_value():
    """
    author: Jiayi
    reviewer: Marthin
    category: edge test
    """
    with pytest.raises(ValueError, match = "Column contains NA values."):
        tab1("onset", outbreak)
    return

def test_column_na():
    """
    make sure the input column does not contain NA values
    """
    with pytest.raises(ValueError, match = "Column contains NA values."):
        tab1("onset", outbreak)
    return

def test_tab1_pattern():
    """
    author: Jiayi
    reviewer: Marthin
    category: pattern test
    """
    result = tab1("age", outbreak)

    # Cumulative Percent must end at ~100 ----
    np.testing.assert_almost_equal(tab1("age", outbreak).iloc[-1]["Cumulative Percent"], 100.0, decimal=1)

    # Cumulative Percent must be non-decreasing ----
    assert result["Cumulative Percent"].is_monotonic_increasing

    return
