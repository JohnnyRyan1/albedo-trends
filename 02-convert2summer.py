#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert monthly mosaics to summer.

"""

# Import packages
import numpy as np
import xarray as xr
import rasterio
import glob
import os

# Define user
user = 'jr555'

# Define filepath
mcd_filepath = '/Volumes/meltwater-mapping_satellite-data/data/MCD43A3/mosaics/'
vnp_filepath = '/Volumes/meltwater-mapping_satellite-data/data/VNP43MA3/mosaics/'
vji_filepath = '/Volumes/meltwater-mapping_satellite-data/data/VJ143MA3/mosaics/'

# Define files
mcd_files = sorted(glob.glob(mcd_filepath + 'monthly/*.tif'))
vnp_files = sorted(glob.glob(vnp_filepath + 'monthly/*.tif'))
vji_files = sorted(glob.glob(vji_filepath + 'monthly/*.tif'))

# Define years
years = np.arange(2000, 2025, 1)

# Define savepath
savepath = '/Users/' + user + '/Desktop/'

#%%

mean_albedo = []
for y in years:
    year_list = []
    for file in mcd_files:
        # Get files for one year
        infileshortname = os.path.basename(file)
        
        if str(y) in infileshortname:
            year_list.append(file)

    # Keep only June, July, August files that do NOT have "_qa"
    summer_files = [f for f in year_list
                    if "_qa" not in f and any(f"-{month:02d}" in f for month in [6, 7, 8])]

    
    if len(summer_files) == 3:

        arrays = []
        
        # Read all files
        for path in summer_files:
            with rasterio.open(path) as src:
                data = src.read(1).astype('float32')
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                arrays.append(data)
                
                # Save metadata from first file
                if 'transform' not in locals():
                    transform = src.transform
                    crs = src.crs
                    height, width = src.height, src.width
        
        # Stack and average across the three months
        stack = np.stack(arrays)
        mean_albedo.append(np.nanmean(stack, axis=0))
    else:
        print("Error...")

# Convert to 3D numpy array: shape (time, y, x)
data_stack = np.stack(mean_albedo)

# Create coordinates from affine transform
x_coords = np.arange(width) * transform.a + transform.c
y_coords = np.arange(height) * transform.e + transform.f

# Create Dataset
ds = xr.Dataset(
    {
        "albedo": (["time", "y", "x"], data_stack)
    },
    coords={
        "time": years,
        "y": y_coords,
        "x": x_coords
    },
    attrs={
        'crs': crs.to_string(),
        "description": "Mean summer albedo",
        "units": "unitless"
    }
)

# Save to NetCDF
ds.to_netcdf(mcd_filepath + "summer/mcd43a3-summer-albedo.nc")


#%%

def process_albedo_series(files, output_nc_path, years, summer_months=(6, 7, 8)):
    mean_albedo = []

    for y in years:
        # Filter files by year and summer months, ignore QA files
        summer_files = [
            f for f in files
            if str(y) in os.path.basename(f)
            and "_qa" not in f
            and any(f"-{month:02d}" in f for month in summer_months)
        ]

        if len(summer_files) == len(summer_months):
            arrays = []

            for path in summer_files:
                with rasterio.open(path) as src:
                    data = src.read(1).astype('float32')
                    nodata = src.nodata
                    if nodata is not None:
                        data[data == nodata] = np.nan
                    arrays.append(data)

                    if 'transform' not in locals():
                        transform = src.transform
                        crs = src.crs
                        height, width = src.height, src.width

            stack = np.stack(arrays)
            mean_albedo.append(np.nanmean(stack, axis=0))
        else:
            print(f"Skipping {y} — expected {len(summer_months)} files, found {len(summer_files)}")

    # Stack into 3D array
    data_stack = np.stack(mean_albedo)

    # Coordinates from transform
    x_coords = np.arange(width) * transform.a + transform.c
    y_coords = np.arange(height) * transform.e + transform.f

    # Create Dataset
    ds = xr.Dataset(
        {
            "albedo": (["time", "y", "x"], data_stack)
        },
        coords={
            "time": years,
            "y": y_coords,
            "x": x_coords
        },
        attrs={
            "crs": crs.to_string(),
            "description": "Mean summer albedo",
            "units": "unitless"
        }
    )

    ds.to_netcdf(output_nc_path)
    print(f"Saved NetCDF: {output_nc_path}")

# Example usage:
process_albedo_series(files=mcd_files, output_nc_path=savepath + 'mcd43a3-summer-albedo.nc',
                      years=np.arange(2000, 2025))

process_albedo_series(files=vnp_files, output_nc_path=savepath + 'vnp43ma3-summer-albedo.nc',
                      years=np.arange(2012, 2025))

process_albedo_series(files=vji_files, output_nc_path=savepath + 'vj143ma3-summer-albedo.nc',
                      years=np.arange(2018, 2025))




#%%















