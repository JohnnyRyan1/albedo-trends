#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

RMSE and bias at each station for the 20198-2024 period.

"""

# Import packages
import pandas as pd
import numpy as np
import xarray as xr
import glob
import os

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'
mask_path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'

# PROMICE
promice_files = sorted(glob.glob(path + 'promice/month/*.csv'))

# Satellite
mcd_points = xr.open_dataset(path + 'satellite/mcd43a3-points.nc')
vnp_points = xr.open_dataset(path + 'satellite/vnp43ma3-points.nc')
vji_points = xr.open_dataset(path + 'satellite/vji43ma3-points.nc')

# Import AWS metadata
aws = pd.read_csv(path + 'promice/AWS_sites_metadata.csv')

# Import climatologies
mcd_clim = xr.open_dataset(path + 'satellite/mcd-albedo-07.tif')
vnp_clim = xr.open_dataset(path + 'satellite/vnp-albedo-07.tif')
vji_clim = xr.open_dataset(path + 'satellite/vji-albedo-07.tif')

# Read mask
ismip_1km = xr.open_dataset(mask_path + '1km-ISMIP6-GIMP.nc')
mask = ismip_1km['GIMP'].values

#%%

def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse

mcd_rmse_values, mcd_bias_values = [], []
elevation_values, latitude_values = [], []
for idx in range(mcd_points['albedo'].shape[1]):

    # Compare albedo at point-scale
    df = pd.DataFrame(list(zip(mcd_points['albedo'][:,idx]['time'].values,
                               mcd_points['albedo'][:,idx].values)), columns=['date', 'sat_albedo'])
    df = df.set_index(['date'])
    
    # Find corresponding AWS data
    filepath = []
    for file in promice_files:
        filename = os.path.basename(file)
        if str(mcd_points['albedo'][:,idx]['aws'].values) + '_' in filename:
            filepath.append(file)
    
    if len(filepath) == 1:
        
        # Append elevation and latitude
        site = aws[aws['site_id'] == str(mcd_points['albedo'][:,idx]['aws'].values)].index
        elevation_values.append(aws.iloc[site]['altitude_last_valid'].values[0])
        latitude_values.append(aws.iloc[site]['latitude_last_valid'].values[0])
        
        aws_df = pd.read_csv(filepath[0], index_col=['time'], parse_dates=['time'])
        
        # Concatenate based on datetime
        df_combined = pd.concat([df, aws_df], axis=1)
        
        # Remove NaNs
        df_combined = df_combined.dropna(subset=['sat_albedo'])
        df_combined = df_combined.dropna(subset=['albedo'])
        
        # Exclude non-summer months
        df_combined = df_combined[df_combined.index.month.isin([6, 7, 8])]
        df_combined = df_combined[df_combined.index > '2018-01-01']
        df_summer = df_combined.resample('Y').mean()
        df_count = df_combined.resample('Y').count()
        df_summer[df_count['sat_albedo'] != 3] = np.nan
        df_summer[df_count['albedo'] != 3] = np.nan

        # Compute stats
        mcd_rmse_values.append(rmse(df_summer['albedo'], df_summer['sat_albedo']))
        mcd_bias_values.append(np.nanmean(df_summer['albedo'] - df_summer['sat_albedo']))
    else:
        print("Error for %.0f" %idx)
        
# Convert to DataFrame
mcd_error_df = pd.DataFrame(list(zip(mcd_rmse_values, mcd_bias_values)), columns=['rmse_mcd', 'bias_mcd'])

vnp_rmse_values, vnp_bias_values = [], []
for idx in range(vnp_points['albedo'].shape[1]):

    # Compare albedo at point-scale
    df = pd.DataFrame(list(zip(vnp_points['albedo'][:,idx]['time'].values,
                               vnp_points['albedo'][:,idx].values)), columns=['date', 'sat_albedo'])
    df = df.set_index(['date'])
    
    # Find corresponding AWS data
    filepath = []
    for file in promice_files:
        filename = os.path.basename(file)
        if str(vnp_points['albedo'][:,idx]['aws'].values) + '_' in filename:
            filepath.append(file)
    
    if len(filepath) == 1:
    
        aws_df = pd.read_csv(filepath[0], index_col=['time'], parse_dates=['time'])
        
        # Concatenate based on datetime
        df_combined = pd.concat([df, aws_df], axis=1)
        
        # Remove NaNs
        df_combined = df_combined.dropna(subset=['sat_albedo'])
        df_combined = df_combined.dropna(subset=['albedo'])
        
        # Exclude non-summer months
        df_combined = df_combined[df_combined.index.month.isin([6, 7, 8])]
        df_combined = df_combined[df_combined.index > '2018-01-01']
        df_summer = df_combined.resample('Y').mean()
        df_count = df_combined.resample('Y').count()
        df_summer[df_count['sat_albedo'] != 3] = np.nan
        df_summer[df_count['albedo'] != 3] = np.nan

        # Compute stats
        vnp_rmse_values.append(rmse(df_summer['albedo'], df_summer['sat_albedo']))
        vnp_bias_values.append(np.nanmean(df_summer['albedo'] - df_summer['sat_albedo']))
    else:
        print("Error for %.0f" %idx)
        
# Convert to DataFrame
vnp_error_df = pd.DataFrame(list(zip(vnp_rmse_values, vnp_bias_values)), columns=['rmse_vnp', 'bias_vnp'])

vji_rmse_values, vji_bias_values = [], []
for idx in range(vji_points['albedo'].shape[1]):

    # Compare albedo at point-scale
    df = pd.DataFrame(list(zip(vji_points['albedo'][:,idx]['time'].values,
                               vji_points['albedo'][:,idx].values)), columns=['date', 'sat_albedo'])
    df = df.set_index(['date'])
    
    # Find corresponding AWS data
    filepath = []
    for file in promice_files:
        filename = os.path.basename(file)
        if str(vji_points['albedo'][:,idx]['aws'].values) + '_' in filename:
            filepath.append(file)
    
    if len(filepath) == 1:
    
        aws_df = pd.read_csv(filepath[0], index_col=['time'], parse_dates=['time'])
        
        # Concatenate based on datetime
        df_combined = pd.concat([df, aws_df], axis=1)
        
        # Remove NaNs
        df_combined = df_combined.dropna(subset=['sat_albedo'])
        df_combined = df_combined.dropna(subset=['albedo'])
        
        # Exclude non-summer months
        df_combined = df_combined[df_combined.index.month.isin([6, 7, 8])]
        df_combined = df_combined[df_combined.index > '2018-01-01']
        df_summer = df_combined.resample('Y').mean()
        df_count = df_combined.resample('Y').count()
        df_summer[df_count['sat_albedo'] != 3] = np.nan
        df_summer[df_count['albedo'] != 3] = np.nan

        # Compute stats
        vji_rmse_values.append(rmse(df_summer['albedo'], df_summer['sat_albedo']))
        vji_bias_values.append(np.nanmean(df_summer['albedo'] - df_summer['sat_albedo']))
    else:
        print("Error for %.0f" %idx)
        
# Convert to DataFrame
vji_error_df = pd.DataFrame(list(zip(vji_rmse_values, vji_bias_values)), columns=['rmse_vji', 'bias_vji'])

# Concatenate
error_df = pd.concat([mcd_error_df, vnp_error_df, vji_error_df], axis=1)

# Add elevation and latitude
error_df['elevation'] = elevation_values
error_df['latitude'] = latitude_values

error_df.to_csv(path + 'station-error.csv', index=False)


#%%








