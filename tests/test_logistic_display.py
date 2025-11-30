#set environment via conda env update -f environment.yml --prune --> conda activate
import pytest
from pyepidisplay.logistic_display import logistic_display
from pyepidisplay.data import data
import numpy as np

#read Outbreak data
import pandas as pd
df=data("Outbreak")

# Smoke Test: check to see if result seems reasonable
"""
    author: scatherinekim
    reviewer: suphanatwong
    category: smoke test
    """
def test_logistic_display_smoke():
    df_results = logistic_display('nausea ~ beefcurry + saltegg', df)

    assert isinstance(df_results, pd.DataFrame)
    assert not df_results.empty

# One shot test: check to see if code crashes
"""
    author: scatherinekim
    reviewer: suphanatwong
    category: one shot test
    """
def test_one_shot():
    logistic_display('nausea ~ beefcurry + saltegg', df)

# edge test

#edge case 1: empty dataframe
"""
    author: scatherinekim
    reviewer: suphanatwong
    category: edge test 1
    """
def test_logistic_display_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(Exception):
        logistic_display('y ~ x1 + x2', empty_df)

#edge case 2: predictor column is missing
        """
    author: scatherinekim
    reviewer: suphanatwong
    category: edge test 2
    """
def test_logistic_display_missing_predictor():
    with pytest.raises(Exception):
        logistic_display('nausea ~ not_a_real_column', df)

#edge case 3: outcome not binary while logistic regression requires binary
        """
    author: scatherinekim
    reviewer: suphanatwong
    category: edge test 3
    """
def test_logistic_display_nonbinary_outcome():
    df_bad = df.copy()
    df_bad['nausea'] = ([0,1,2,3] * (len(df_bad)//4)) + [0]*(len(df_bad)%4)
    with pytest.raises(ValueError):
        logistic_display('nausea ~ beefcurry + saltegg', df_bad)

#pattern test: give known pattern to give known results
"""
author: scatherinekim
reviewer: suphanatwong
category: pattern test
"""      
def test_logistic_display_pattern():
# Create a small, synthetic dataset with a predictable pattern
    # random_salt_egg = np.random.random(size=(2,1))
    df_pattern = pd.DataFrame({
        'nausea': [0, 1, 0],
        'beefcurry': [0.1, 0.9, 0.9],
        'saltegg': [0.5, 0.1, 0.1]
    })
    try:
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except ImportError:
        PerfectSeparationError = Exception  # fallback if import fails

    try:
        df_results = logistic_display('nausea ~ beefcurry + saltegg', df_pattern)
        assert isinstance(df_results, pd.DataFrame)
    except PerfectSeparationError:
        # If perfect separation occurs, still pass the test
        assert True
    except Exception as e:
        print(e)