# Tests for ci_prop

from pyepidisplay.ci_prop import ci_prop
import warnings
import numpy as np
import pytest

# Smoke Test for ci_prop
def test_smoke():
    """
    Simple smoke test to make sure the function runs and returns a dict.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ci_prop([1])

    assert isinstance(result, dict)
    return

# Edge Test for ci_prop
def test_all_zeros():
    """
    If all values are 0, proportion should be 0, CI should be valid numbers.
    """
    result = ci_prop([0, 0, 0, 0])
    assert isinstance(result, dict)
    assert result["proportion"] == 0


def test_all_ones():
    """
    If all values are 1, proportion should be 1.
    """
    result = ci_prop([1, 1, 1])
    assert isinstance(result, dict)
    assert result["proportion"] == 1


def test_mixed_values():
    """
    Mixed 0/1 values should compute a valid proportion.
    """
    result = ci_prop([0, 1, 1, 0, 1])
    assert isinstance(result, dict)
    assert 0 <= result["proportion"] <= 1


def test_empty_list_returns_nan():
    """
    Empty list should return NaNs (suppress warnings).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ci_prop([])

    assert isinstance(result, dict)
    assert np.isnan(result["proportion"])
    assert np.isnan(result["ci_lower"])
    assert np.isnan(result["ci_upper"])


def test_non_binary_values_raise_typeerror():
    """
    ci_prop should raise TypeError for invalid values.
    """
    with pytest.raises(TypeError):
        ci_prop(["katze", 1])    # wrong type


def test_single_value():
    """
    Single value returns a valid dict with 0 or 1.
    CI may be NaN or computed depending on implementation.
    """
    result = ci_prop([1])
    assert isinstance(result, dict)
    assert result["proportion"] == 1
