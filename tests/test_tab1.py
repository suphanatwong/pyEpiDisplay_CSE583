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
    Simple smoke test to make sure function runs.
    """
    tab1("age", outbreak)
    return

def test_column_validity():
    """
    make sure the input column is in valid data type
    """
    with pytest.raises(ValueError, match ="Column name must be a string."):
        tab1("age", outbreak)
    return

def test_column_existence():
    """
    make sure the input column exists in the dataframe
    """
    with pytest.raises(ValueError, match = "Column is not found in DataFrame."):
        tab1("ages", outbreak)
    return

def test_column_na():
    """
    make sure the input column does not contain NA values
    """
    with pytest.raises(ValueError, match = "Column contains NA values."):
        tab1("age", outbreak)
    return

def test_dataframe_validity():
    """
    make sure the input dataframe is in valid data type
    """
    with pytest.raises(ValueError, match = "Input data must be a pandas DataFrame."):
        tab1("age", outbreak)
    return