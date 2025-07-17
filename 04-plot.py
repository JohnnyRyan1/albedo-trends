#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Box plot

"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Import data
df = pd.read_csv(path + 'station-error.csv')


#%%

"""
Box plot

"""

# Drop rows with any missing RMSE or bias values
metrics_df = df[['rmse_mcd', 'rmse_vnp', 'rmse_vji',
                 'bias_mcd', 'bias_vnp', 'bias_vji']].dropna()

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

# RMSE boxplot
metrics_df[['rmse_mcd', 'rmse_vnp', 'rmse_vji']].boxplot(ax=axes[0], showfliers=False)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_xticklabels(['MCD43A3', 'VNP43MA3', 'VJI43MA3'], fontsize=12)
axes[0].grid(True)

# Bias boxplot
metrics_df[['bias_mcd', 'bias_vnp', 'bias_vji']].boxplot(ax=axes[1], showfliers=False)
axes[1].set_ylabel('Bias', fontsize=12)
axes[1].set_xticklabels(['MCD43A3', 'VNP43MA3', 'VJI43MA3'], fontsize=12)
axes[1].grid(True)

#%%


"""

By elevation and latitude

"""

# Set up subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 4), layout='constrained')







#%%





#%%


