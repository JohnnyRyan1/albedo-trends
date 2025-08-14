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
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pymannkendall as mk

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define files
files = sorted(glob.glob(path + 'station/*.csv'))

#%%
p_values = []
r_values = []
station = []
residual_trend = []
residual_sig = []
rmse_linear = []

for file in files:
    
    # Get station name
    s = os.path.basename(file)[:-4]
    
    # Read file
    df = pd.read_csv(file, index_col=['datetime'], parse_dates=['datetime'])
    df = df[2:]
    
    # Compute differences between AWS and MCD
    mask = np.isfinite(df['aws']) & np.isfinite(df['mcd'])
    if np.sum(np.isfinite(df['aws'])) > 5:
        
        # Append
        station.append(s)
        
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
        
        # Prepare the predictor matrix
        X = df[['t2m', 'sf_summer', 'sf_winter']]

        y = df['mcd']
        y.name = 'albedo'

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Shuffle and split: 20 train, 3 test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=3, random_state=1)

        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and calculate residuals
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Compute residuals
        residuals = y_train - y_pred

        # Calculate RMSE
        rmse_linear.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        
        # Append
        residual_trend.append(mk.original_test(residuals).trend)
        residual_sig.append(mk.original_test(residuals).p)

#%%

"""

Determine primary drivers of albedo.

"""


p_df = pd.DataFrame(p_values, columns=['t2m', 'sf_winter', 'sf_summer'])
r_df = pd.DataFrame(r_values, columns=['t2m', 'sf_winter', 'sf_summer'])

print('Mean r between albedo and summer air temperature is %.2f' % (np.mean(r_df['t2m'])))
print('Mean r between albedo and summer snwofall is %.2f' % (np.mean(r_df['sf_summer'])))
print('Mean r between albedo and winter snowfall is %.2f' % (np.mean(r_df['sf_winter'])))

#%%

"""

Investigate if albedo is decreasing after removing dependence on summer air temperature and snowfall.

"""

residual_df = pd.DataFrame(list(zip(station, residual_trend)), columns=['station', 'residual_trend'])

#%%

"""

Benchmark a multiple linear regression for predicting albedo.

"""

linear_df = pd.DataFrame(list(zip(station, rmse_linear)), columns=['station', 'linear'])

#%%

"""

Investigate if sensitivity of albedo is correlated with air temperatures.

"""

cool_mask = df['t2m'] < np.median(df['t2m'])
warm_mask = df['t2m'] >= np.median(df['t2m'])

# Regress albedo vs. temp in each bin
model_cool = LinearRegression().fit(df['t2m'][cool_mask].values.reshape(-1, 1), df['mcd'][cool_mask])
model_warm = LinearRegression().fit(df['t2m'][warm_mask].values.reshape(-1, 1), df['mcd'][warm_mask])

print(f"Slope (cool temps): {model_cool.coef_[0]:.3f}")
print(f"Slope (warm temps): {model_warm.coef_[0]:.3f}")

#%%












