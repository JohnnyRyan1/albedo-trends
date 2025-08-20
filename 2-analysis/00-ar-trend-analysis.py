#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test different orders of autoregressive models.

"""

# Import packages
import pandas as pd
import numpy as np
import glob
from statsmodels.tsa.ar_model import AutoReg

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
hindcast_files = sorted(glob.glob(path + 'data/hindcast/*.csv'))
modern_files = sorted(glob.glob(path + 'data/station/*.csv'))

n_list = np.arange(0,6,1)

best_n_hindcast = []
best_n_modern = []
for f in range(len(hindcast_files)):

    # Read data
    hindcast_df = pd.read_csv(hindcast_files[f], index_col=['datetime'], parse_dates=['datetime'])
    modern_df = pd.read_csv(modern_files[f], index_col=['datetime'], parse_dates=['datetime'])

    bic_hindcast = []
    bic_modern = []
    for n in n_list:
    
        ar0_hindcast = AutoReg(hindcast_df['y_pred_lin'], lags=n, trend='ct', seasonal=False)
        results_hindcast = ar0_hindcast.fit()
        bic_hindcast.append(results_hindcast.bic)
        
        ar0_modern = AutoReg(modern_df['mcd'], lags=n, trend='ct', seasonal=False)
        results_modern = ar0_modern.fit()
        bic_modern.append(results_modern.bic)
        
    best_n_hindcast.append(np.array(bic_hindcast).argmin())
    best_n_modern.append(np.array(bic_modern).argmin())
    
#%%

print(np.sum(np.array(best_n_modern) == 0)/len(best_n_modern))
print(np.sum(np.array(best_n_hindcast) == 0)/len(best_n_hindcast))

#%%










