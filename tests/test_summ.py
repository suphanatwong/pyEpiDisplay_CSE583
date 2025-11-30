#Test function for summ

# test_summ_function.py

import pytest
import pandas as pd
import numpy as np
from CSE583_summ_function import summ

def test_one_shot():
    """
    author: Marthin
    reviewer: Jiayi
    category: one shot test
    """
    data = [1, 2, 3, 4]
    result = summ(pd.Series(data))
    np.testing.assert_allclose(result["mean"], 2.5)  # expected mean
    return





def test_smoke():
    """
    author: Marthin
    reviewer: Jiayi
    category: smoke test
    """
    summ([0, 1, 2])  # should run without error
    return

#testing
pytest summ_testfunc.py



#Edge test
          """
    author: Marthin
    reviewer: Jiayi
    category: edge test
    """

def test_edge_invalid_input():
    """
    author: Marthin
    reviewer: Jiayi
    category: edge test
    """
    with pytest.raises(ValueError, match="Input must be numeric"):
        summ(["a", "b", "c"])
        return




#Pattern test

def test_pattern_alternating_values():
    """
    author: Marthin
    reviewer: Jiayi
    category: pattern test
    """
    series = pd.Series([1, 2, 1, 2, 1, 2])
    result = summ(series)
    np.testing.assert_allclose(result["mean"], 1.5)
    np.testing.assert_allclose(result["median"], 1.5)
    return
