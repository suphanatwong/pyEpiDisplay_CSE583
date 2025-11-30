#set environment via conda env update -f environment.yml --prune --> conda activate
import pytest
from pyepidisplay.logistic_display import logistic_display
from pyepidisplay.data import data
import numpy as np

#read Outbreak data
import pandas as pd
df=data("Outbreak")

# Smoke Test: check to see if result seems reasonable
"""
    author: scatherinekim
    reviewer: 
    category: smoke test
    """