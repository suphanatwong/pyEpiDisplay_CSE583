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

# In[3]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

def regress_display(model):
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
    outcome = model.model.data.ynames
    print(f"Linear regression predicting {outcome}\n")

    coef = model.params
    ci = model.conf_int()
    ci.columns = ["Lower 95% CI", "Upper 95% CI"]
    p_t = model.pvalues

    # Compute sequential Type I ANOVA F-test p-values
    anova_table = sm.stats.anova_lm(model, typ=1)

    data = []
    for var in coef.index:
        if var == 'Intercept':
            continue

        # Label binary variables
        if var in model.model.data.orig_exog.columns:
            unique_vals = model.model.data.orig_exog[var].dropna().unique()
            if set(unique_vals) <= {0, 1}:
                label = f"{var}: 1 vs 0"
            else:
                label = var
        else:
            label = var

        p_f = anova_table.loc[var, "PR(>F)"] if var in anova_table.index else np.nan

        data.append({
            "Variable": label,
            "adj coef (95% CI)": f"{coef[var]:.2f} ({ci.loc[var,'Lower 95% CI']:.2f}, {ci.loc[var,'Upper 95% CI']:.2f})",
            "P(t-test)": round(p_t[var], 3),
            "P(F-test)": round(p_f, 3)
        })

    table = pd.DataFrame(data)
    print(f"No. of observations = {int(model.nobs)}\n")
    return table


#test function

#read Outbreak data
import pandas as pd
df = pd.read_csv('Outbreak.csv')

import statsmodels.formula.api as smf
model = smf.ols('onset ~ beefcurry + saltegg', data=df).fit()
df_results = regress_display(model)
print(df_results) #need to troubleshoot
#add lower 95% CI

