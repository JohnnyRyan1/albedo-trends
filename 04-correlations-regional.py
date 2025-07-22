#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute correlation coefficients between albedo, air temperature, and snowfall

"""

# Import packages
import pandas as pd
from scipy.stats import pearsonr
import numpy as np


# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define files
t2m = pd.read_csv(path + 'era5/t2m.csv', index_col=['datetime'], parse_dates=True)
sf_w = pd.read_csv(path + 'era5/sf_w.csv', index_col=['datetime'], parse_dates=True)
sf_s = pd.read_csv(path + 'era5/sf_s.csv', index_col=['datetime'], parse_dates=True)
mcd = pd.read_csv(path + 'satellite/mcd43a3.csv', index_col=0, parse_dates=True)
mcd = mcd[mcd.index.month.isin([6, 7, 8])]
mcd_summer = mcd.resample('YE').mean()
mcd_count = mcd.resample('YE').count()
mcd_summer[mcd_count != 3] = np.nan

t2m = t2m[t2m.index.isin(mcd_summer.index)]
sf_w = sf_w[sf_w.index.isin(mcd_summer.index)]
sf_s = sf_s[sf_s.index.isin(mcd_summer.index)]

#%%

# Initialize result containers
r_values = {}
p_values = {}

# Assume both dataframes have the same column names and order
for col in mcd_summer.columns:
    r, p = pearsonr(t2m[col], mcd_summer[col])
    r_values[col] = r
    p_values[col] = p

# Convert to Series or DataFrame if needed
r_series = pd.Series(r_values, name='r')
p_series = pd.Series(p_values, name='p')


#%%

p_df = pd.DataFrame(p_values, columns=['t2m', 'sf_winter', 'sf_summer'])
r_df = pd.DataFrame(r_values, columns=['t2m', 'sf_winter', 'sf_summer'])
















