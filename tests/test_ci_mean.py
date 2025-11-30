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
