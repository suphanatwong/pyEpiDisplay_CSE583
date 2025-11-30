#This is for the project itself, for the HW 3 I did ci_prop. Anna 
# Smoke Test for ci_mean
from pyepidisplay.ci_mean import ci_mean

"""
I am still working on this. For HW3, I did ci_prop.
"""

def test_smoke():
    """
    Simple smoke test to make sure the function runs and returns a dict.
    """
    result = ci_mean([1])
    assert isinstance(result, dict)
    return

# Edge Tests for ci_mean

def test_negative_values_allowed():
    """
    ci_mean allows negative numbers; just check it returns a dict.
    """
    result = ci_mean([-5, 10, 20])
    assert isinstance(result, dict)
    assert "mean" in result


def test_empty_list_returns_dict_with_nan():
    """
    ci_mean([]) returns dict with NaNs.
    Suppress warnings locally inside this test.
    """
    import warnings
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ci_mean([])

    assert isinstance(result, dict)
    assert np.isnan(result["mean"])
    assert np.isnan(result["sd"])
    assert np.isnan(result["se"])
    assert np.isnan(result["ci_lower"])
    assert np.isnan(result["ci_upper"])


def test_single_value_returns_dict_with_nan_se_ci():
    """
    ci_mean([single_value]) returns mean but se/CI are NaN.
    Suppress warnings locally inside this test.
    """
    import warnings
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ci_mean([42])

    assert isinstance(result, dict)
    assert result["mean"] == 42
    assert np.isnan(result["se"])
    assert np.isnan(result["ci_lower"])
    assert np.isnan(result["ci_upper"])


def test_non_numeric_input_raises_typeerror():
    """
    ci_mean should raise TypeError for non-numeric values.
    """
    import pytest

    with pytest.raises(TypeError):
        ci_mean([1, "a", 3])

# One shot test 
def test_outbreak_dataset_matches_r_output():
    """
    Validate ci_mean against known R output from outbreak dataset.
    This one-shot test verifies the function produces correct results.
    

    """
    import pandas as pd
    import numpy as np
    
    # Load outbreak data
    outbreak_df = pd.read_csv("src/pyepidisplay/datasets/Outbreak.csv")
    result = ci_mean(outbreak_df["age"])
    
    # Expected values from R's epiDisplay
    assert np.isclose(result["mean"], 23.69104, atol=1e-4)
    assert np.isclose(result["sd"], 19.67349, atol=1e-4)
    assert np.isclose(result["se"], 0.5948025, atol=1e-4)
    assert np.isclose(result["ci_lower"], 22.52396, atol=1e-4)
    assert np.isclose(result["ci_upper"], 24.85813, atol=1e-4)

# Pattern test: 
def test_pattern_se_decreases_with_sample_size():
    """
    Pattern test: As sample size increases with the same population,
    the standard error (SE) should decrease.
    This follows the pattern that larger samples provide more precise estimates.
    """
    import numpy as np
    
    se_values = []
    
    # Loop over increasing sample sizes, repeating the same data
    base_data = [20, 22, 25, 19, 21]  # Small fixed dataset
    for n in [5, 10, 15, 20]:
        # Repeat the base data to create sample of size n
        sample = (base_data * (n // len(base_data) + 1))[:n]
        result = ci_mean(sample)
        se_values.append(result["se"])
        
        # Verify result is valid
        assert isinstance(result, dict)
        assert not np.isnan(result["se"])
        assert not np.isnan(result["ci_lower"])
        assert not np.isnan(result["ci_upper"])
    
    # Verify pattern: SE should be decreasing (or equal, never increasing)
    for i in range(len(se_values) - 1):
        assert se_values[i] >= se_values[i + 1], \
            f"SE should decrease with sample size: {se_values}"
