#!/usr/bin/env python
# coding: utf-8

# ## Replicating ci.prop() function from R into Python

import numpy as np
from scipy.stats import norm

def ci_prop(x, n=None, ci=0.95):
    # Case 1: x is a list/array of 0/1 data
    if n is None:
        x = np.array(x)
        x = x[~np.isnan(x)]          # remove NA
        n = len(x)
        x = np.sum(x)                # count successes

    # Case 2: x is number of successes, n is provided
    p = x / n
    se = np.sqrt(p * (1 - p) / n)

    alpha = 1 - ci
    z = norm.ppf(1 - alpha / 2)

    lower = p - z * se
    upper = p + z * se

    # Keep between 0 and 1
    lower = max(lower, 0)
    upper = min(upper, 1)

    return {
        "proportion": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
        "x": x
    }


# ## Pretty Printer

def print_ci_prop(res, ci=0.95):
    print(f"Proportion:   {res['proportion']:.4f}")
    print(f"SE:           {res['se']:.4f}")
    print(f"{int(ci*100)}% CI:    ({res['ci_lower']:.4f}, {res['ci_upper']:.4f})")


# ## Test on the outbreak dataset

import pandas as pd

outbreak_df = pd.read_csv("outbreak.csv")
outbreak_df.head()

print_ci_prop(ci_prop(outbreak_df["beefcurry"]))


# ## Comparrison with R

# $proportion
# [1] 0.9533821
# 
# $se
# [1] 0.006373841
# 
# $ci_lower
# [1] 0.9408896
# 
# $ci_upper
# [1] 0.9658746
# 
# $n
# [1] 1094
# 
# $x
# [1] 1043

# Numbers match

