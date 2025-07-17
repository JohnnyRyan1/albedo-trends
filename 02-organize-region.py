#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Export ice sheet and regional albedo means.

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
mask = ismip_1km['GIMP'].values
elevation = ismip_1km['SRF'].values
mar_mask = xr.open_dataset(savepath + 'mar-masks.nc')

# Read regions
regions_file = xr.open_dataset(mask_path + 'temp_albedo_summer_climatologies.nc')
regions = regions_file['regions'].values

# Define files
mcd_files = sorted(glob.glob(mcd_filepath + '*.tif'))
vnp_files = sorted(glob.glob(vnp_filepath + '*.tif'))
vji_files = sorted(glob.glob(vji_filepath + '*.tif'))

def compute_albedo_stats(data, mask, elevation, regions):
    """Compute average albedo for full ice sheet, elevation bands, and regions."""
    stats_mean = [np.nanmean(data[mask]),
                  np.nanmean(data[mask & (elevation < 2000)]),
                  np.nanmean(data[mask & (elevation > 2000)])]
    
    stats_std = [np.nanstd(data[mask]),
                  np.nanstd(data[mask & (elevation < 2000)]),
                  np.nanstd(data[mask & (elevation > 2000)])]
    
    for r in range(1, 9):
        reg_mask = mask & (regions == r)
        stats_mean += [np.nanmean(data[reg_mask]),
                       np.nanmean(data[reg_mask & (elevation < 2000)]),
                       np.nanmean(data[reg_mask & (elevation > 2000)])]
        
        stats_std += [np.nanstd(data[reg_mask]),
                      np.nanstd(data[reg_mask & (elevation < 2000)]),
                      np.nanstd(data[reg_mask & (elevation > 2000)])]
    
    return stats_mean, stats_std

def process_albedo_files(file_list, mask, elevation, regions):
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

        stats_mean, stats_std = compute_albedo_stats(data, mask, elevation, regions)

        if 'qa' in filename:
            stats_qa.append(stats_mean)
            stats_qa_std.append(stats_std)
            dates_qa.append(dt)
        else:
            stats_main.append(stats_mean)
            stats_main_std.append(stats_std)
            dates_main.append(dt)

    # Column labels
    cols = ['ice_sheet', 'ice_sheet_below', 'ice_sheet_above'] + \
           [f'region{r}_{sfx}' for r in range(1, 9) for sfx in ['full', 'below', 'above']]
    
    # Create DataFrames
    main_df = pd.DataFrame(stats_main, columns=cols, index=dates_main)
    main_std_df = pd.DataFrame(stats_main_std, columns=cols, index=dates_main)
    qa_df   = pd.DataFrame(stats_qa,   columns=cols, index=dates_qa)
    qa_std_df   = pd.DataFrame(stats_qa_std,   columns=cols, index=dates_qa)

    return main_df, qa_df, main_std_df, qa_std_df

#%%
# Process each set
mcd_df, mcd_std_df, mcd_qa_df, mcd_qa_std_df = process_albedo_files(mcd_files, mask, elevation, regions)
vnp_df, vnp_std_df, vnp_qa_df, vnp_qa_std_df = process_albedo_files(vnp_files, mask, elevation, regions)
vji_df, vji_std_df, vji_qa_df, vji_qa_std_df = process_albedo_files(vji_files, mask, elevation, regions)

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

# Compute climatologies

months = ['04', '05', '06', '07', '08', '09', '10']

# Loop through each GeoTIFF and read data
for month in months:
    arrays, arrays_qa = [], []
    for file in mcd_files:
        filename = os.path.basename(file)
        if month == filename[13:15]:
            if 'qa' in filename:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays_qa.append(data)
            else:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays.append(data)    
    
    # Stack and compute mean across the stack
    stack = np.stack(arrays)
    mean_albedo = np.nanmean(stack, axis=0)
    
    stack_qa = np.stack(arrays_qa)
    mean_albedo_qa = np.nanmean(stack_qa, axis=0)
    
    # Save the result to a new GeoTIFF
    out_path = os.path.join(savepath, 'satellite/mcd-albedo-' + month + '.tif')
    with rasterio.open(mcd_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo.astype('float32'), 1)
    
    out_path = os.path.join(savepath, 'satellite/mcd-albedo-' + month + '-qa.tif')
    with rasterio.open(mcd_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo_qa.astype('float32'), 1)
    
    print(f"Mean albedo written to: {out_path}")


#%%


# Compute climatologies

months = ['04', '05', '06', '07', '08', '09', '10']

# Loop through each GeoTIFF and read data
for month in months:
    arrays, arrays_qa = [], []
    for file in vnp_files:
        filename = os.path.basename(file)
        if month == filename[14:16]:
            if 'qa' in filename:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays_qa.append(data)
            else:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays.append(data)    
    
    # Stack and compute mean across the stack
    stack = np.stack(arrays)
    mean_albedo = np.nanmean(stack, axis=0)
    
    stack_qa = np.stack(arrays_qa)
    mean_albedo_qa = np.nanmean(stack_qa, axis=0)
    
    # Save the result to a new GeoTIFF
    out_path = os.path.join(savepath, 'satellite/vnp-albedo-' + month + '.tif')
    with rasterio.open(vnp_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo.astype('float32'), 1)
    
    out_path = os.path.join(savepath, 'satellite/vnp-albedo-' + month + '-qa.tif')
    with rasterio.open(vnp_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo_qa.astype('float32'), 1)
    
    print(f"Mean albedo written to: {out_path}")


#%%


# Compute climatologies

months = ['04', '05', '06', '07', '08', '09', '10']

# Loop through each GeoTIFF and read data
for month in months:
    arrays, arrays_qa = [], []
    for file in vji_files:
        filename = os.path.basename(file)
        if month == filename[14:16]:
            if 'qa' in filename:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays_qa.append(data)
            else:
                src = rasterio.open(file) 
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Mask out nodata values
                if nodata is not None:
                    data[data == nodata] = np.nan

                arrays.append(data)    
    
    # Stack and compute mean across the stack
    stack = np.stack(arrays)
    mean_albedo = np.nanmean(stack, axis=0)
    
    stack_qa = np.stack(arrays_qa)
    mean_albedo_qa = np.nanmean(stack_qa, axis=0)
    
    # Save the result to a new GeoTIFF
    out_path = os.path.join(savepath, 'satellite/vji-albedo-' + month + '.tif')
    with rasterio.open(vji_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo.astype('float32'), 1)
    
    out_path = os.path.join(savepath, 'satellite/vji-albedo-' + month + '-qa.tif')
    with rasterio.open(vji_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1, nodata=np.nan)
    
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(mean_albedo_qa.astype('float32'), 1)
    
    print(f"Mean albedo written to: {out_path}")


#%%














