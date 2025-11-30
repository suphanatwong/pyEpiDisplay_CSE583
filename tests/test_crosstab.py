# test_my_crosstab.py

import pytest
import pandas as pd
from CSE583_Crosstab_Function import my_crosstab   # adjust filename if needed

def test_one_shot(capsys):
    """One-shot test: verify output structure on known input."""
    # Create a simple dataset
    x = pd.Series(["yes", "no", "yes", "no", "yes"])
    y = pd.Series(["A", "A", "B", "B", "B"])
    
    # Run function
    my_crosstab(x, y, chisq=True)
    
    # Capture printed output
    captured = capsys.readouterr()
    
    # Check that key sections are present in output
    assert "Counts:" in captured.out
    assert "Row Percentages" in captured.out
    assert "Column Percentages" in captured.out
    assert "Chi-square Test" in captured.out

def test_smoke():
    """Smoke test: ensure function runs without crashing."""
    x = pd.Series(["cat", "dog", "cat", "dog"])
    y = pd.Series(["male", "female", "male", "female"])
    
    try:
        my_crosstab(x, y, chisq=False)  # run without chi-square
    except Exception as e:
        pytest.fail(f"Smoke test failed with exception: {e}")



#Testing

pytest crosstab_testfunc.py



#Edge test
def test_edge_empty_inputs(capsys):
    x = pd.Series([])
    y = pd.Series([])
    my_crosstab(x, y, chisq=False)
    captured = capsys.readouterr()
    assert "Counts:" in captured.out

def test_edge_single_category(capsys):
    x = pd.Series(["yes", "yes", "yes"])
    y = pd.Series(["A", "A", "A"])
    my_crosstab(x, y, chisq=False)
    captured = capsys.readouterr()
    assert "Counts:" in captured.out




#Pattern test
def test_pattern_alternating_categories(capsys):
    x = pd.Series(["yes", "no", "yes", "no"])
    y = pd.Series(["A", "A", "B", "B"])
    my_crosstab(x, y, chisq=True)
    captured = capsys.readouterr()
    assert "Chi-square Test" in captured.out

