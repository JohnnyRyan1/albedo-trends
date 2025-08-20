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
user = 'jr555'

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
predicted_albedo, observed_albedo = [], []

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
        predicted_albedo.append(np.mean(y_pred_test))
        observed_albedo.append(np.mean(y_train))

#%%

"""
Save predicted vs. observed albedo

"""

stats_df = pd.DataFrame(list(zip(station, observed_albedo, predicted_albedo,rmse_linear)),
                        columns=['station', 'observed', 'predicted', 'rmse'])
stats_df.to_csv(path + 'linear-model.csv')

print(stats_df['rmse'].mean())
print(stats_df['rmse'].mean() / stats_df['observed'].mean())

#%%

"""

Determine primary drivers of albedo.

"""

climate_df = pd.concat([
    pd.DataFrame(station, columns=['station']),
    pd.DataFrame(p_values, columns=['t2m_p', 'sf_winter_p', 'sf_summer_p']),
    pd.DataFrame(r_values, columns=['t2m_r', 'sf_winter_r', 'sf_summer_r'])
], axis=1)

climate_df.to_csv(path + 'climate-correlations.csv')


print('Mean r between albedo and summer air temperature is %.2f' % (np.mean(climate_df['t2m_r'])))
print('Mean r between albedo and summer snwofall is %.2f' % (np.mean(climate_df['sf_summer_r'])))
print('Mean r between albedo and winter snowfall is %.2f' % (np.mean(climate_df['sf_winter_r'])))

#%%

"""

Investigate if albedo is decreasing after removing dependence on summer air temperature and snowfall.

"""

residual_df = pd.DataFrame(list(zip(station, residual_trend, residual_sig)), 
                           columns=['station', 'residual_trend', 'residual_sig'])

residual_df.to_csv(path + 'residual.csv')

#%%












