#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute ice sheet and regional climate means.

"""

# Import packages
import numpy as np
import pandas as pd
import xarray as xr

# Define user
user = 'johnnyryan'

# Define filepath
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Read mask
ismip_1km = xr.open_dataset(path1 + '1km-ISMIP6-GIMP.nc')
ice_mask = ismip_1km['GIMP'].values
mar_mask = xr.open_dataset(path2 + 'mar-masks.nc')
smb_mask = mar_mask['smb'].values
melt_mask = mar_mask['melt'].values

# Read regions
regions_file = xr.open_dataset(path1 + 'temp_albedo_summer_climatologies.nc')
regions = regions_file['regions'].values

# Define ERA5 data
era_t2m = xr.open_dataset(path2 + 'era5/era-summer-t2m-1941-2025.nc')
era_summer_sf = xr.open_dataset(path2 + 'era5/era-summer-sf-1941-2025.nc')
era_winter_sf = xr.open_dataset(path2 + 'era5/era-winter-sf-1941-2025.nc')


def compute_stats(data, ice_mask, smb_mask, melt_mask, regions):
    """
    Compute mean climate data for full ice sheet, ablation zone, 
    wet snow zone, dry snow zone, and regions.
    """
    stats_mean = [np.nanmean(data.where([ice_mask]), axis=(1,2)),
                  np.nanmean(data.where([ice_mask & (smb_mask < 0)]), axis=(1,2)),
                  np.nanmean(data.where([ice_mask & (smb_mask > 0) & (melt_mask == 0)]), axis=(1,2)),
                  np.nanmean(data.where([ice_mask & (smb_mask > 0) & (melt_mask > 0)]), axis=(1,2))]
    
    stats_std = [np.nanstd(data.where([ice_mask]), axis=(1,2)),
                  np.nanstd(data.where([ice_mask & (smb_mask < 0)]), axis=(1,2)),
                  np.nanstd(data.where([ice_mask & (smb_mask > 0) & (melt_mask == 0)]), axis=(1,2)),
                  np.nanstd(data.where([ice_mask & (smb_mask > 0) & (melt_mask > 0)]), axis=(1,2))]
    
    for r in range(1, 9):
        reg_mask = ice_mask & (regions == r)
        stats_mean += [np.nanmean(data.where([reg_mask]), axis=(1,2)),
                      np.nanmean(data.where([reg_mask & (smb_mask < 0)]), axis=(1,2)),
                      np.nanmean(data.where([reg_mask & (smb_mask > 0) & (melt_mask == 0)]), axis=(1,2)),
                      np.nanmean(data.where([reg_mask & (smb_mask > 0) & (melt_mask > 0)]), axis=(1,2))]
        
        stats_std += [np.nanstd(data.where([reg_mask]), axis=(1,2)),
                      np.nanstd(data.where([reg_mask & (smb_mask < 0)]), axis=(1,2)),
                      np.nanstd(data.where([reg_mask & (smb_mask > 0) & (melt_mask == 0)]), axis=(1,2)),
                      np.nanstd(data.where([reg_mask & (smb_mask > 0) & (melt_mask > 0)]), axis=(1,2))]


    # Column labels
    cols = ['ice_sheet', 'ablation_zone', 'wet_snow_zone', 'dry_snow_zone'] + \
           [f'region{r}_{sfx}' for r in range(1, 9) for sfx in ['full', 'abl', 'wet', 'dry']]
    
    # Define dates
    dates_main = pd.to_datetime([f"{year}-12-31" for year in data['year'].values])
    
    # Create DataFrames
    main_df = pd.DataFrame(stats_mean).T
    main_df.columns = cols
    main_df.index = dates_main
    
    main_std_df = pd.DataFrame(stats_std).T
    main_std_df.columns = cols
    main_df.index = dates_main

    return main_df, main_std_df

# Process each set
t2m_df, t2m_std_df = compute_stats(era_t2m['t2m'], ice_mask, smb_mask, melt_mask, regions)
sf_w_df, sf_w_std_df = compute_stats(era_winter_sf['sf'], ice_mask, smb_mask, melt_mask, regions)
sf_s_df, sf_s_std_df = compute_stats(era_summer_sf['sf'], ice_mask, smb_mask, melt_mask, regions)

# Rename index
t2m_df.index.name = 'datetime'
t2m_std_df.index.name = 'datetime'
sf_w_df.index.name = 'datetime'
sf_w_std_df.index.name = 'datetime'
sf_s_df.index.name = 'datetime'
sf_s_std_df.index.name = 'datetime'

# Save as csv
t2m_df.to_csv(path2 + 'era5/t2m.csv')
t2m_std_df.to_csv(path2 + 'era5/t2m_std.csv')
sf_w_df.to_csv(path2 + 'era5/sf_w.csv')
sf_w_std_df.to_csv(path2 + 'era5/sf_w_std.csv')
sf_s_df.to_csv(path2 + 'era5/sf_s.csv')
sf_s_std_df.to_csv(path2 + 'era5/sf_s_std.csv')

#%%

