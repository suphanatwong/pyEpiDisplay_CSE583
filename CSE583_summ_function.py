#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing pandas and the EpiDisplay folder
import pandas as pd
import numpy as np
import pyreadr
import os
import matplotlib.pyplot as plt
import seaborn as sns

#path to .rdata file
file_path = "/Users/marthinmandig/Downloads/epiDisplay/data/Outbreak.rdata"

#read rdata file
result = pyreadr.read_r(file_path)

#check objects inside
print(result.keys())


# In[2]:


#Extract dataset

df = result['Outbreak']

#Preview dataset
print(df.head())


# # Recreating summ() function in python

# In[ ]:


#summ() function = Summary of data frame in a convenient table. Summary of a variable with statistics and graph. Includes the mean, median, standard deviation, min, and max.
#Must compare two or more variables

#Creating the function
def summ(series):
    clean_series = series.dropna()
    summary = {
        "obs": clean_series.count(),
        "mean": clean_series.mean(),
        "median": clean_series.median(),
        "s.d.": clean_series.std(),
        "min": clean_series.min(),
        "max": clean_series.max()
    }
    return summary


# Summary of age for males 13+ with nausea
subset = df[(df['sex'] == 1) & (df['age'] >= 13) & (df['age'] != 99) & (df['nausea'] == 1)]
print(summ(subset['age']))


# In[5]:


#Creating the visualization aspect (Not part of the function but a way to see the comparison)

# Boxplot
sns.boxplot(x=subset['age'])
plt.title("Age Distribution of Males (13+) with Nausea")
plt.xlabel("Age (years)")
plt.show()

grouped = df[df['age'] != 99].groupby('sex')['age'].apply(summ)
print(grouped)

