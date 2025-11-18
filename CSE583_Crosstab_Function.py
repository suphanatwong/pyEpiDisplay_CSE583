#!/usr/bin/env python
# coding: utf-8

# Documentation:
# 
# 
#     General-purpose cross-tabulation function.
# 
#     This function generates a contingency table of counts between two categorical variables,
#     along with row percentages, column percentages, and an optional chi-square test of independence.

# In[5]:


#Import

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import pyreadr
import os
import matplotlib.pyplot as plt
import seaborn as sns

#path to .rdata file
file_path = "/Users/marthinmandig/Downloads/epiDisplay/data/Outbreak.rdata"

#read rdata file
result = pyreadr.read_r(file_path)

df = result['Outbreak']


# In[7]:


def my_crosstab(x, y, chisq=True):
    """
    General-purpose cross-tabulation function.
    Displays counts, row percentages, column percentages,
    and optionally runs a chi-square test.
    
    Parameters:
    x : pandas Series (outcome variable)
    y : pandas Series (exposure/grouping variable)
    chisq : bool (default True) - run chi-square test
    """
    
    # Create contingency table
    tab = pd.crosstab(x, y, dropna=False)
    
    print("\nCounts:")
    print(tab)
    
    print("\nRow Percentages (%):")
    row_pct = tab.div(tab.sum(axis=1), axis=0) * 100
    print(row_pct.round(1))
    
    print("\nColumn Percentages (%):")
    col_pct = tab.div(tab.sum(axis=0), axis=1) * 100
    print(col_pct.round(1))
    
    if chisq:
        chi2, p, dof, expected = chi2_contingency(tab)
        print("\nChi-square Test:")
        print(f"Chi2 = {chi2:.3f}, df = {dof}, p-value = {p:.4f}")
        print("\nExpected counts:")
        print(pd.DataFrame(expected, 
                           index=tab.index, 
                           columns=tab.columns).round(1))


# In[8]:


# Suppose df is your Outbreak dataset loaded via pyreadr
# Crosstab nausea vs sex
my_crosstab(df['nausea'], df['sex'])


# In[ ]:


jupyter nbconvert --to script 'CSE583_Crosstab_Function.ipynb'

