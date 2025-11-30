#!/usr/bin/env python
# coding: utf-8

# The logistic.display() function in R takes in a fitted logistic regression object and outputs a logistic model summary. When the Outbreak Investigations dataset is the dataset, this function can be used to model the probability that a person experiences a certain symptom such as nausea as a function of whether they ate a certain food such as beef curry.
# 
# The output provides the following data: 
# - The crude OR(95% CI) is the unadjusted odds ratio obtained from modeling each predictor alone with the outcome.
# - The adj. OR(95% CI) is the adjusted odds ratio from the multivariate model that includes all predictors simultaenously.
# - P(Wald's test) is the Wald test p-value that evaluates whether the coefficient (log-odds) for that variable differs from 0.
# - P(LR-test) is the likelihood ratio p value that compares the full model vs a model without that predictor.
# - The output also provides model fit statistics like the log-likelihood, the number of observations, and the AIC (Akaike Information Criterion).


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

def logistic_display(formula, data):
    """
    Mimics epiDisplay::logistic.display in R.
    Computes both crude and adjusted ORs, 95% CIs, and Wald p-values.

    Parameters:
        formula (str): e.g. 'nausea ~ beefcurry + saltegg'
        data (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: Crude ORs, Adjusted ORs, p-values (Wald test), and LR test p-values
    """

    # Parse outcome and predictors
    outcome, rhs = formula.split("~")
    outcome = outcome.strip()
    predictors = [x.strip() for x in rhs.split("+")]

    # Fit adjusted (multivariate) model
    full_model = smf.logit(formula=formula, data=data).fit(disp=0)
    ll_full = full_model.llf
    adj_params = full_model.params
    adj_ci = full_model.conf_int()
    adj_pvals = full_model.pvalues

    results = []

    for predictor in predictors:
        #crude model (univariate)
        crude_model = smf.logit(f"{outcome} ~ {predictor}", data=data).fit(disp=0)
        crude_params = crude_model.params[predictor]
        crude_ci = np.exp(crude_model.conf_int().loc[predictor])
        crude_OR = np.exp(crude_params)

        #adjusted model
        adj_params_pred = adj_params[predictor]
        adj_OR = np.exp(adj_params_pred)
        adj_ci_pred = np.exp(adj_ci.loc[predictor])

        #wald p-value
        p_wald = adj_pvals[predictor]

        #LR test p-p value
        reduced_predictors = [v for v in predictors if v != predictor]

        # if removing the variable leaves no predictors, use intercept-only model
        if reduced_predictors:
            reduced_formula = outcome + " ~ " + " + ".join(reduced_predictors)
        else:
            reduced_formula = outcome + " ~ 1"

        reduced_model = smf.logit(reduced_formula, data=data).fit(disp=0)
        ll_reduced = reduced_model.llf

        lr_stat = 2 * (ll_full - ll_reduced)
        p_lr = 1 - stats.chi2.cdf(lr_stat, df=1)

        results.append({
            "Variable": predictor,
            "Crude OR (95% CI)": f"{crude_OR:.2f} ({crude_ci[0]:.2f}, {crude_ci[1]:.2f})",
            "Adj. OR (95% CI)": f"{adj_OR:.2f} ({adj_ci_pred[0]:.2f}, {adj_ci_pred[1]:.2f})",
            "P(Wald)": f"{p_wald:.2f}",
            "P(LR-test)": f"{p_lr:.2f}"
        })

    # Print model summary info
    print(f"\nLog-likelihood = {full_model.llf:.4f}")
    print(f"No. of observations = {int(full_model.nobs)}")
    print(f"AIC value = {full_model.aic:.4f}\n")

    return pd.DataFrame(results)

    # Smoke Test
#read Outbreak data
import pandas as pd

def test_logistic_display_smoke():
    df = pd.read_csv('/home/stlp/pyEpiDisplay/src/pyEpiDisplay/datasets')

    df_results = logistic_display('nausea ~ beefcurry + saltegg', df)

    assert isinstance(df_results, pd.DataFrame)
    assert not df_results.empty

    print(df_results)