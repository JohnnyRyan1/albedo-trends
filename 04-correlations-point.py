#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute correlation coefficients between albedo, air temperature, and snowfall

"""

# Import packages
import pandas as pd
from scipy.stats import linregress
import numpy as np
import glob
import matplotlib.pyplot as plt


# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define files
files = sorted(glob.glob(path + 'station/*.csv'))


#%%
p_values = []
r_values = []

for file in files:
    # Read file
    df = pd.read_csv(file, index_col=['datetime'])
    
    # Mask
    x = df['t2m'][(np.isfinite(df['t2m']) & (np.isfinite(df['mcd'])))]
    y = df['mcd'][(np.isfinite(df['t2m']) & (np.isfinite(df['mcd'])))]
        
    # Compute stats
    slope1, intercept1, r1, p1, se1 = linregress(x, y)
    
    # Mask
    x = df['sf_winter'][(np.isfinite(df['sf_winter']) & (np.isfinite(df['mcd'])))]
    y = df['mcd'][(np.isfinite(df['sf_winter']) & (np.isfinite(df['mcd'])))]
        
    # Compute stats
    slope2, intercept2, r2, p2, se2 = linregress(x, y)
    
    # Mask
    x = df['sf_summer'][(np.isfinite(df['sf_summer']) & (np.isfinite(df['mcd'])))]
    y = df['mcd'][(np.isfinite(df['sf_summer']) & (np.isfinite(df['mcd'])))]
        
    # Compute stats
    slope3, intercept3, r3, p3, se3 = linregress(x, y)
    
    # Append
    p_values.append([p1, p2, p3])
    r_values.append([r1, r2, r3])

#%%

p_df = pd.DataFrame(p_values, columns=['t2m', 'sf_winter', 'sf_summer'])
r_df = pd.DataFrame(r_values, columns=['t2m', 'sf_winter', 'sf_summer'])
















