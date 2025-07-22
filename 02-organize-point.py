#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Export AWS albedo data.

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

# AWS locations
aws_meta = pd.read_csv(savepath +'/promice/AWS_sites_metadata.csv')
aws_meta = aws_meta[aws_meta['location_type'] == 'ice sheet']
point_ids = aws_meta['site_id'].values

# Define conversion factors
conversion_factors = xr.open_dataset(savepath + 'satellite/conversion-factors.nc')

# Define files
mcd_files = sorted(glob.glob(mcd_filepath + '*.tif'))
vnp_files = sorted(glob.glob(vnp_filepath + '*.tif'))
vji_files = sorted(glob.glob(vji_filepath + '*.tif'))

lons, lats = [], []
for a in range(len(aws_meta)):
    # Find lat and lon
    lat, lon = aws_meta.iloc[a]['latitude_last_valid'], aws_meta.iloc[a]['longitude_last_valid']
    
    dist_sq = (ismip_1km['lat'].values - lat)**2 + (ismip_1km['lon'].values - lon)**2
    i, j = np.unravel_index(np.argmin(dist_sq), ismip_1km['lat'].values.shape)
    lons.append(j)
    lats.append(i)

#%%
def extract_albedo_to_netcdf(files, lons, lats, point_ids, out_prefix):
    """Extract albedo time series from files and save as NetCDF (main + QA)."""
    date = []
    date_qa = []
    values = []
    values_qa = []

    for file in files:
        print(f"Processing: {file}")
        filename = os.path.basename(file)

        # Parse timestamp
        if filename[0] == 'M':
            year = filename[8:12]
            month = filename[13:15]
        else:
            year = filename[9:13]
            month = filename[14:16]
        dt = pd.to_datetime(f"{year}-{month}-01") + pd.offsets.MonthEnd(1)

        # Read raster data
        with rasterio.open(file) as src:
            data = src.read(1)

        # Sample values at given lat/lon indices
        point_values = [data[lat, lon] for lat, lon in zip(lats, lons)]

        # Append to the appropriate list
        if 'qa' in filename:
            date_qa.append(dt)
            values_qa.append(point_values)
        else:
            date.append(dt)
            values.append(point_values)

    # Convert to DataArrays
    da_main = xr.DataArray(
        data=np.array(values),
        coords={"time": pd.to_datetime(date), "aws": point_ids},
        dims=["time", "aws"],
        name="albedo"
    )
    da_qa = xr.DataArray(
        data=np.array(values_qa),
        coords={"time": pd.to_datetime(date_qa), "aws": point_ids},
        dims=["time", "aws"],
        name="albedo_qa"
    )

    # Save both as NetCDFs
    da_main.to_dataset().to_netcdf(f"{out_prefix}-points.nc")
    da_qa.to_dataset().to_netcdf(f"{out_prefix}-qa-points.nc")
    print(f"Saved: {out_prefix}-points.nc and {out_prefix}-qa-points.nc")

# Use on MCD43A3
extract_albedo_to_netcdf(mcd_files, lons, lats, point_ids, savepath + 'sattelite/mcd43a3')

# Use on VNP43MA3
extract_albedo_to_netcdf(vnp_files, lons, lats, point_ids, savepath + 'sattelite/vnp43ma3')

# Use on VJI (optional)
extract_albedo_to_netcdf(vji_files, lons, lats, point_ids, savepath + 'sattelite/vji43ma3')

#%%





















