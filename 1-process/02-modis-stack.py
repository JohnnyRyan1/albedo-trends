#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

1. Read and stack MCD43A3/VJ143MA3/VNP43MA3 HDFs

"""

# Import modules
import rioxarray as rio
import xarray as xr
import numpy as np
import os
import glob
from datetime import datetime, timedelta

#%%

# Define paths
path1 = '/Volumes/meltwater-mapping_satellite-data/data/'
path2 = '/Users/jr555/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'

# Define products
products = ['MCD43A3', 'VJ143MA3', 'VNP43MA3']

# Define tiles
tiles = ['h15v01', 'h15v02', 'h16v00', 'h16v01', 'h16v02', 'h17v00', 'h17v01', 'h17v02']

# Import surface water raster for matching
ismip = rio.open_rasterio(path2 + '1km-ISMIP6-GIMP.tif')
ismip = ismip.rio.write_crs('EPSG:3413')


#%%

def day_of_year_to_month(day_of_year, year):

    # Start from January 1st of the given year
    start_of_year = datetime(year, 1, 1)
    
    # Add the day_of_year offset (subtracting 1 because day_of_year starts at 1)
    target_date = start_of_year + timedelta(days=day_of_year - 1)
    
    # Return the month
    return target_date.month

#%%

""" 
Count files
"""

count = []
for tile in tiles:
    
    # Define location of MODIS data
    files = sorted(glob.glob(path1 + 'VNP43MA3/' + tile + '/*.h5'))

    # Append
    count.append(len(files))
    
# Define years
years = np.arange(2000, 2025, 1)

# Define months
months = np.arange(4, 10, 1)

# Define all files
all_files = sorted(glob.glob(path1 + 'VJ143MA3/*/*.h5'))

#%%
years = [2019]
months = [6]
for year in years:
    for month in months:
        mean_albedo_list, mean_albedo_qa_list = [], []
        for tile in tiles:
            # Make a list of files
            modis_files_list = []
            for file in all_files:
                
                # Get the path and filename separately
                infilepath, infilename = os.path.split(file)
                
                # Get the short name (filename without extension)
                infileshortname, extension = os.path.splitext(infilename)
                
                # Define day of year
                day = infileshortname[13:16]
                
                # Define month
                month_file = day_of_year_to_month(int(day), year)
                
                if (infileshortname[9:13] == str(year)) &\
                   (month_file == month) &\
                   (infileshortname[18:23] == tile):
                    modis_files_list.append(file)

            # Produce mean albedo
            data_arrays = []
            quality_arrays = []
            for file in modis_files_list:
                
                # Read
                modis_data = rio.open_rasterio(file)
                
                # Get snow albedo
                sw_albedo = modis_data["Albedo_BSA_shortwave"]
                sw_quality_flag = modis_data["BRDF_Albedo_Band_Mandatory_Quality_shortwave"]
                
                # Remove size-1 dimension
                sw_albedo = sw_albedo.squeeze(dim="band")
                sw_quality_flag = sw_quality_flag.squeeze(dim="band")
                
                data_arrays.append(sw_albedo)
                quality_arrays.append(sw_quality_flag)
            
            # Stack the DataArrays along a new dimension (e.g., 'time')
            stacked_da = xr.concat(data_arrays, dim="time")
            stacked_qa = xr.concat(quality_arrays, dim="time") 
            
            # Set values equal to 32767 to NaN
            stacked_da = stacked_da.where(stacked_da != 32767, np.nan)
            
            # Scale
            stacked_da = stacked_da*stacked_da.scale_factor
                        
            # Set values greater than 1 to NaN
            stacked_da = stacked_da.where(stacked_da <= 1, np.nan)
            
            # Set values less than 0.2 to NaN
            stacked_da = stacked_da.where(stacked_da >= 0.2, np.nan)
            
            # Set values that don't have good quality flag to NaN
            stacked_da_aq = stacked_da.where(stacked_qa == 0, np.nan)

            # Compute nanmean along the time dimension
            mean_albedo = stacked_da.mean(dim="time", skipna=True)
            mean_albedo_qa = stacked_da_aq.mean(dim="time", skipna=True)
            
            # Append
            mean_albedo_list.append(mean_albedo)
            mean_albedo_qa_list.append(mean_albedo_qa)

        # Merge the DataArrays into one
        merged_da = xr.combine_by_coords(mean_albedo_list)
        merged_da_qa = xr.combine_by_coords(mean_albedo_qa_list)
        
        # Reproject to EPSG:3413
        merged_da = merged_da.rio.reproject("EPSG:3413")
        merged_da_qa = merged_da_qa.rio.reproject("EPSG:3413")

        # Match to surface water extent 500m
        merged_da_match = merged_da.rio.reproject_match(ismip)
        merged_da_qa_match = merged_da_qa.rio.reproject_match(ismip)
        
        # Export as GeoTiff
        merged_da_match['Albedo_BSA_shortwave'].rio.to_raster(path + infileshortname[0:7] + '-' + str(year) + '-' + str(month).zfill(2) + '.tif')
        merged_da_qa_match['Albedo_BSA_shortwave'].rio.to_raster(path + infileshortname[0:7] + '-' + str(year) + '-' + str(month).zfill(2) + '_qa.tif')


#%%















































