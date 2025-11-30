#Test function for summ

# test_summ_function.py

import pytest
import pandas as pd
from CSE583_summ_function import summ

def test_one_shot():
    """One-shot test: check a known input/output pair."""
    data = [1, 2, 3, 4]
    result = summ(data)
    assert result == 10  # expected sum

def test_smoke():
    """Smoke test: just verify the function runs without errors."""
    try:
        _ = summ([0, 1, 2])
    except Exception as e:
        pytest.fail(f"Smoke test failed with exception: {e}")



#testing
pytest summ_testfunc.py



#Edge test
def test_edge_empty_series():
    series = pd.Series([])
    result = summ(series)
    assert result["obs"] == 0
    assert pd.isna(result["mean"])

def test_edge_all_nan():
    series = pd.Series([None, None, None])
    result = summ(series)
    assert result["obs"] == 0
    assert pd.isna(result["mean"])

def test_edge_single_value():
    series = pd.Series([42])
    result = summ(series)
    assert result["obs"] == 1
    assert result["mean"] == 42
    assert result["median"] == 42
    assert result["min"] == 42
    assert result["max"] == 42



#Pattern test

def test_pattern_alternating_values():
    series = pd.Series([1, 2, 1, 2, 1, 2])
    result = summ(series)
    assert result["mean"] == pytest.approx(1.5)
    assert result["median"] == 1.5
