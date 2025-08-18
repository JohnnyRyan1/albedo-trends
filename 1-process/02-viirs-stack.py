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
products = ['VNP43MA3']

# Define tiles
tiles = ['h15v01', 'h15v02', 'h16v00', 'h16v01', 'h16v02', 'h17v00', 'h17v01', 'h17v02']

# Import surface water raster for matching
ismip = rio.open_rasterio(path2 + '1km-ISMIP6-GIMP.tif')
ismip = ismip.rio.write_crs('EPSG:3413')

def day_of_year_to_month(day_of_year, year):

    # Start from January 1st of the given year
    start_of_year = datetime(year, 1, 1)
    
    # Add the day_of_year offset (subtracting 1 because day_of_year starts at 1)
    target_date = start_of_year + timedelta(days=day_of_year - 1)
    
    # Return the month
    return target_date.month
   
# Define years
years = np.arange(2012, 2018, 1)

# Define months
months = np.arange(4, 9, 1)

for product in products:
    print(product)
    # Define all files
    all_files = sorted(glob.glob(path1 + product + '/*/*.h5'))
    for year in years:
        print('Year = %s' %year)
        for month in months:
            if os.path.exists(path1 + product + '/mosaics/' + product + '-' + str(year) + '-' + str(month).zfill(2) + '.tif'):
                pass
            else:
                print('Month = %s' %month)
                mean_albedo_list, mean_albedo_qa_list = [], []
                for tile in tiles:
                    # Make a list of files
                    viirs_files_list = []
                    for file in all_files:
                        
                        # Get the path and filename separately
                        infilepath, infilename = os.path.split(file)
                        
                        # Get the short name (filename without extension)
                        infileshortname, extension = os.path.splitext(infilename)
                        
                        # Define day of year
                        day = infileshortname[14:17]
                        
                        # Define month
                        month_file = day_of_year_to_month(int(day), year)
                        
                        if (infileshortname[10:14] == str(year)) &\
                           (month_file == month) &\
                           (infileshortname[18:24] == tile):
                            viirs_files_list.append(file)
                    
                    # Produce mean albedo
                    data_arrays = []
                    quality_arrays = []
                    for file in viirs_files_list:
                        
                        # Read
                        viirs_data = rio.open_rasterio(file)
                        
                        # Get snow albedo
                        field = 'HDFEOS_GRIDS_VIIRS_Grid_BRDF_Data_Fields_'
                        sw_albedo = viirs_data[field + 'Albedo_BSA_shortwave']
                        sw_quality_flag = viirs_data[field + "BRDF_Albedo_Band_Mandatory_Quality_shortwave"]
                        
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
                    stacked_da_qa = stacked_da.where(stacked_qa == 0, np.nan)
        
                    # Compute nanmean along the time dimension
                    mean_albedo = stacked_da.mean(dim="time", skipna=True)
                    mean_albedo_qa = stacked_da_qa.mean(dim="time", skipna=True)
                    
                    # Append
                    mean_albedo_list.append(mean_albedo)
                    mean_albedo_qa_list.append(mean_albedo_qa)
        
                # Merge the DataArrays into one
                merged_da = xr.combine_by_coords(mean_albedo_list)
                merged_da_qa = xr.combine_by_coords(mean_albedo_qa_list)
                
                # Write projection
                if product == 'VNP43MA3':
                    merged_da = merged_da.rio.write_crs(viirs_data.spatial_ref.attrs['crs_wkt'])
                    merged_da_qa = merged_da_qa.rio.write_crs(viirs_data.spatial_ref.attrs['crs_wkt'])
                else:
                    merged_da = merged_da.rio.write_crs(viirs_data['Projection'].attrs['crs_wkt'])
                    merged_da_qa = merged_da_qa.rio.write_crs(viirs_data['Projection'].attrs['crs_wkt'])
        
                # Reproject to EPSG:3413
                merged_da = merged_da.rio.reproject("EPSG:3413")
                merged_da_qa = merged_da_qa.rio.reproject("EPSG:3413")
        
                # Match to surface water extent 1 km
                merged_da_match = merged_da.rio.reproject_match(ismip)
                merged_da_qa_match = merged_da_qa.rio.reproject_match(ismip)
                
                # Export as GeoTiff
                merged_da_match[field + 'Albedo_BSA_shortwave'].rio.to_raster(path1 + product + '/mosaics/' + product + '-' + str(year) + '-' + str(month).zfill(2) + '.tif')
                merged_da_qa_match[field + 'Albedo_BSA_shortwave'].rio.to_raster(path1 + product + '/mosaics/'  + product + '-' + str(year) + '-' + str(month).zfill(2) + '_qa.tif')
    
    
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





























