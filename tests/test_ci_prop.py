# Tests for ci_prop

from pyepidisplay.ci_prop import ci_prop
import warnings
import numpy as np
import pytest

# Smoke Test for ci_prop
def test_smoke():
    """
    Simple smoke test to make sure the function runs and returns a dict.
    
    author: Anna
    reviewer: Cat
    category: smoke test
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
    
    author: Anna
    reviewer: Cat
    category: edge test
    """
    result = ci_prop([0, 0, 0, 0])
    assert isinstance(result, dict)
    assert result["proportion"] == 0


def test_all_ones():
    """
    If all values are 1, proportion should be 1.
    
    author: Anna
    reviewer: Cat
    category: edge test
    """
    result = ci_prop([1, 1, 1])
    assert isinstance(result, dict)
    assert result["proportion"] == 1


def test_mixed_values():
    """
    Mixed 0/1 values should compute a valid proportion.
    
    author: Anna
    reviewer: Cat
    category: edge test
    """
    result = ci_prop([0, 1, 1, 0, 1])
    assert isinstance(result, dict)
    assert 0 <= result["proportion"] <= 1


def test_empty_list_returns_nan():
    """
    Empty list should return NaNs (suppress warnings).
    
    author: Anna
    reviewer: Cat
    category: edge test
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
    
    author: Anna
    reviewer: Cat
    category: edge test
    """
    with pytest.raises(TypeError):
        ci_prop(["katze", 1])    # wrong type


def test_single_value():
    """
    Single value returns a valid dict with 0 or 1.
    CI may be NaN or computed depending on implementation.
    
    author: Anna
    reviewer: Cat
    category: edge test
    """
    result = ci_prop([1])
    assert isinstance(result, dict)
    assert result["proportion"] == 1

def test_one_shot():
    """
    author: Anna
    reviewer: Cat
    category: one-shot test
    """
    x = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    result = ci_prop(x)

    # Check fields
    assert("proportion" in result)
    assert("se" in result)
    assert("ci_lower" in result)
    assert("ci_upper" in result)
    assert("n" in result)
    assert("x" in result)

    assert(result["proportion"] == 0.7)
    assert(result["se"] == np.sqrt(0.7 * 0.3 / 10))
    assert(np.isclose(result["ci_lower"], 0.4160, atol=1e-4))
    assert(np.isclose(result["ci_upper"], 0.9840, atol=1e-4))
    assert(result["n"] == 10)
    assert(result["x"] == 7)

def test_pattern_ci_width_decreases_with_sample_size():
    """
    Pattern test: As sample size increases with the same proportion,
    the confidence interval width should decrease (narrower CI).
    This follows the pattern that larger samples provide more precise estimates.
    
    author: Anna
    reviewer: Cat
    category: pattern test
    """
    ci_widths = []
    
    # Loop over different sample sizes with constant proportion of 0.5
    for n in [10, 50, 100, 500]:
        # Create data with proportion 0.5 (equal 0s and 1s)
        x = [1] * (n // 2) + [0] * (n // 2)
        result = ci_prop(x)
        
        # Calculate CI width
        ci_width = result["ci_upper"] - result["ci_lower"]
        ci_widths.append(ci_width)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert np.isclose(result["proportion"], 0.5, atol=0.01)
        assert result["n"] == n
    
    # Verify pattern: CI widths should be decreasing (or equal, never increasing)
    for i in range(len(ci_widths) - 1):
        assert ci_widths[i] >= ci_widths[i + 1], \
            f"CI width should decrease with sample size: {ci_widths}"