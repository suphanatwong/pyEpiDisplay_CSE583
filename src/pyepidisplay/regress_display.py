#!/usr/bin/env python
# coding: utf-8

# The regress.display() function in R takes in a linear regression object and outputs a table of linear model summary. When the Outbreak Investigations dataset is the dataset, this function can show how the average onset time of showing symptoms changes depending on whether a person ate beef curry or salt egg.
# 
# 
# The output provides the following data: 
# - The adj. coeff
# - The 95% CI
# - P(t-test)
# - P(F-test)
# - No. of observations

import pandas as pd
import numpy as np

def regress_display(model, logistic=False, digits=3):
    """
    Mimics epiDisplay::regress.display in R.

    Parameters
    ----------
    model : fitted statsmodels model
        e.g., smf.ols(...).fit() or smf.logit(...).fit()
    logistic : bool
        Set True for logistic regression to show Odds Ratios instead of Coefficients.
    digits : int
        Number of decimal places to round.
    """
    # Extract outcome variable (works for OLS)
    outcome = model.model.data.ynames

    # Print header
    print(f"Linear regression predicting {outcome}\n")
    coef = model.params
    se = model.bse
    pvals = model.pvalues
    ci = model.conf_int()
    ci.columns = ["Lower 95% CI", "Upper 95% CI"]

    if logistic:
        # Convert β to OR and CI to OR CI
        OR = np.exp(coef)
        OR_ci_low = np.exp(ci["Lower 95% CI"])
        OR_ci_high = np.exp(ci["Upper 95% CI"])

        table = pd.DataFrame({
            "OR": OR.round(digits),
            "Lower 95% CI": OR_ci_low.round(digits),
            "Upper 95% CI": OR_ci_high.round(digits),
            "p-value": pvals.round(digits)
        })

    else:
        # Linear regression output
        table = pd.DataFrame({
            "Coef": coef.round(digits),
            "SE": se.round(digits),
            "Lower 95% CI": ci["Lower 95% CI"].round(digits),
            "Upper 95% CI": ci["Upper 95% CI"].round(digits),
            "p-value": pvals.round(digits)
        })

    return table

#test function

#read Outbreak data
import pandas as pd
df = pd.read_csv('Outbreak.csv')

import statsmodels.formula.api as smf
model = smf.ols('onset ~ beefcurry + saltegg', data=df).fit()
df_results = regress_display(model)
print(df_results)