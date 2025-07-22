#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute ice sheet and regional albedo means.

"""

# Import packages
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import glob
import os

# Define user
user = 'jr555'

# Define filepath
mcd_filepath = '/Volumes/meltwater-mapping_satellite-data/data/MCD43A3/mosaics/'
vnp_filepath = '/Volumes/meltwater-mapping_satellite-data/data/VNP43MA3/mosaics/'
vji_filepath = '/Volumes/meltwater-mapping_satellite-data/data/VJ143MA3/mosaics/'
mask_path = '/Users/jr555/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'
savepath = '/Users/jr555/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Read mask
ismip_1km = xr.open_dataset(mask_path + '1km-ISMIP6-GIMP.nc')
ice_mask = ismip_1km['GIMP'].values
mar_mask = xr.open_dataset(savepath + 'mar-masks.nc')
smb_mask = mar_mask['smb'].values
melt_mask = mar_mask['melt'].values

# Read regions
regions_file = xr.open_dataset(mask_path + 'temp_albedo_summer_climatologies.nc')
regions = regions_file['regions'].values

# Define files
mcd_files = sorted(glob.glob(mcd_filepath + 'monthly/*.tif'))
vnp_files = sorted(glob.glob(vnp_filepath + 'monthly/*.tif'))
vji_files = sorted(glob.glob(vji_filepath + 'monthly/*.tif'))

def compute_albedo_stats(data, ice_mask, smb_mask, melt_mask, regions):
    """
    Compute average albedo for full ice sheet, ablation zone, 
    wet snow zone, dry snow zone, and regions.
    """
    stats_mean = [np.nanmean(data[ice_mask]),
                  np.nanmean(data[ice_mask & (smb_mask < 0)]),
                  np.nanmean(data[ice_mask & (smb_mask > 0) & (melt_mask == 0)]),
                  np.nanmean(data[ice_mask & (smb_mask > 0) & (melt_mask > 0)])]
    
    stats_std = [np.nanstd(data[ice_mask]),
                  np.nanstd(data[ice_mask & (smb_mask < 0)]),
                  np.nanstd(data[ice_mask & (smb_mask > 0) & (melt_mask == 0)]),
                  np.nanstd(data[ice_mask & (smb_mask > 0) & (melt_mask > 0)])]
    
    for r in range(1, 9):
        reg_mask = ice_mask & (regions == r)
        stats_mean += [np.nanmean(data[reg_mask]),
                      np.nanmean(data[reg_mask & (smb_mask < 0)]),
                      np.nanmean(data[reg_mask & (smb_mask > 0) & (melt_mask == 0)]),
                      np.nanmean(data[reg_mask & (smb_mask > 0) & (melt_mask > 0)])]
        
        stats_std += [np.nanstd(data[reg_mask]),
                      np.nanstd(data[reg_mask & (smb_mask < 0)]),
                      np.nanstd(data[reg_mask & (smb_mask > 0) & (melt_mask == 0)]),
                      np.nanstd(data[reg_mask & (smb_mask > 0) & (melt_mask > 0)])]
    
    return stats_mean, stats_std

def process_albedo_files(file_list, ice_mask, smb_mask, melt_mask, regions):
    """Process a list of albedo files and return DataFrames for main and QA products."""
    stats_main, stats_qa = [], []
    dates_main, dates_qa = [], []
    stats_main_std, stats_qa_std = [], []

    for file in file_list:
        print(f"Processing: {file}")
        filename = os.path.basename(file)
        if filename[0] =='M':
            year = filename[8:12]
            month = filename[13:15]
            dt = pd.to_datetime(f"{year}-{month}-01")
        else: 
            year = filename[9:13]
            month = filename[14:16]
            dt = pd.to_datetime(f"{year}-{month}-01")

        with rasterio.open(file) as src:
            data = src.read(1)

        stats_mean, stats_std = compute_albedo_stats(data, ice_mask, smb_mask, melt_mask, regions)

        if 'qa' in filename:
            stats_qa.append(stats_mean)
            stats_qa_std.append(stats_std)
            dates_qa.append(dt)
        else:
            stats_main.append(stats_mean)
            stats_main_std.append(stats_std)
            dates_main.append(dt)

    # Column labels
    cols = ['ice_sheet', 'ablation_zone', 'wet_snow_zone', 'dry_snow_zone'] + \
           [f'region{r}_{sfx}' for r in range(1, 9) for sfx in ['full', 'abl', 'wet', 'dry']]
    
    # Create DataFrames
    main_df = pd.DataFrame(stats_main, columns=cols, index=dates_main)
    main_std_df = pd.DataFrame(stats_main_std, columns=cols, index=dates_main)
    qa_df   = pd.DataFrame(stats_qa,   columns=cols, index=dates_qa)
    qa_std_df   = pd.DataFrame(stats_qa_std,   columns=cols, index=dates_qa)

    return main_df, qa_df, main_std_df, qa_std_df

#%%
# Process each set
mcd_df, mcd_std_df, mcd_qa_df, mcd_qa_std_df = process_albedo_files(mcd_files, ice_mask, smb_mask, melt_mask, regions)
vnp_df, vnp_std_df, vnp_qa_df, vnp_qa_std_df = process_albedo_files(vnp_files, ice_mask, smb_mask, melt_mask, regions)
vji_df, vji_std_df, vji_qa_df, vji_qa_std_df = process_albedo_files(vji_files, ice_mask, smb_mask, melt_mask, regions)

# Save as csv
mcd_df.to_csv(savepath + 'satellite/mcd43a3.csv')
mcd_std_df.to_csv(savepath + 'satellite/mcd43a3_std.csv')
mcd_qa_df.to_csv(savepath + 'satellite/mcd43a3_qa.csv')
mcd_qa_std_df.to_csv(savepath + 'satellite/mcd43a3_qa_std.csv')

vnp_df.to_csv(savepath + 'satellite/vnp43ma3.csv')
vnp_std_df.to_csv(savepath + 'satellite/vnp43ma3_std.csv')
vnp_qa_df.to_csv(savepath + 'satellite/vnp43ma3_qa.csv')
vnp_qa_std_df.to_csv(savepath + 'satellite/vnp43ma3_qa_std.csv')

vji_df.to_csv(savepath + 'satellite/vji43ma3.csv')
vji_std_df.to_csv(savepath + 'satellite/vji43ma3_std.csv')
vji_qa_df.to_csv(savepath + 'satellite/vji43ma3_qa.csv')
vji_qa_std_df.to_csv(savepath + 'satellite/vji43ma3_qa_std.csv')

#%%

